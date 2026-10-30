#include "Preprocessor.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/photo.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/photo/cuda.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>
#include <vector>

namespace exam_hacker {
namespace {

cv::Mat toGray(const cv::Mat& image) {
    if (image.channels() == 1) {
        return image;
    }
    if (image.channels() == 3) {
        cv::Mat gray;
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        return gray;
    }
    throw std::invalid_argument("Preprocessor accepts 1-channel grayscale or 3-channel BGR images");
}

std::array<cv::Point2f, 4> orderCorners(const std::vector<cv::Point>& contour) {
    std::array<cv::Point2f, 4> ordered{};
    float minSum = std::numeric_limits<float>::max();
    float maxSum = std::numeric_limits<float>::lowest();
    float minDiff = std::numeric_limits<float>::max();
    float maxDiff = std::numeric_limits<float>::lowest();
    for (const auto& point : contour) {
        const float sum = static_cast<float>(point.x + point.y);
        const float diff = static_cast<float>(point.x - point.y);
        if (sum < minSum) { minSum = sum; ordered[0] = point; } // top-left
        if (sum > maxSum) { maxSum = sum; ordered[2] = point; } // bottom-right
        if (diff > maxDiff) { maxDiff = diff; ordered[1] = point; } // top-right
        if (diff < minDiff) { minDiff = diff; ordered[3] = point; } // bottom-left
    }
    return ordered;
}

double horizontalProjectionScore(const cv::Mat& binary) {
    cv::Mat rowProjection;
    cv::reduce(binary, rowProjection, 1, cv::REDUCE_SUM, CV_64F);
    const double total = cv::sum(rowProjection)[0];
    if (total <= 0.0) {
        return 0.0;
    }
    // Horizontal text lines concentrate ink into fewer rows. Normalize by the
    // total ink count so interpolation at different angles is comparable.
    return cv::sum(rowProjection.mul(rowProjection))[0] / (total * total);
}

} // namespace

Preprocessor::Preprocessor(PreprocessorOptions options) : options_(std::move(options)) {
    if (options_.maxDeskewDegrees < 0.0 || options_.maxDeskewDegrees > 45.0) {
        throw std::invalid_argument("maxDeskewDegrees must be between 0 and 45");
    }
    if (options_.denoiseStrength < 0.0F) {
        throw std::invalid_argument("denoiseStrength cannot be negative");
    }
    if (options_.claheClipLimit <= 0.0 || options_.claheTileGrid.width <= 0 ||
        options_.claheTileGrid.height <= 0) {
        throw std::invalid_argument("CLAHE clip limit and tile dimensions must be positive");
    }

    if (options_.useCuda) {
        try {
            cudaEnabled_ = cv::cuda::getCudaEnabledDeviceCount() > 0;
            if (cudaEnabled_) {
                cv::cuda::setDevice(0);
            }
        } catch (const cv::Exception&) {
            cudaEnabled_ = false;
        }
    }
}

cv::Mat Preprocessor::process(const cv::Mat& input, PreprocessDiagnostics* diagnostics) const {
    if (input.empty()) {
        throw std::invalid_argument("Cannot preprocess an empty image");
    }
    if (input.channels() != 1 && input.channels() != 3) {
        throw std::invalid_argument("Preprocessor accepts 1-channel grayscale or 3-channel BGR images");
    }

    const bool captureStageImages = diagnostics != nullptr && diagnostics->captureStageImages;
    PreprocessDiagnostics localDiagnostics;
    localDiagnostics.cudaEnabled = cudaEnabled_;
    localDiagnostics.captureStageImages = captureStageImages;
    cv::Mat output = input.clone();
    if (captureStageImages) {
        localDiagnostics.stages.push_back({"input", output.clone()});
    }
    if (options_.rectifyPage) {
        output = rectifyPage(output, &localDiagnostics);
    }
    if (captureStageImages) {
        localDiagnostics.stages.push_back({"page_rectified", output.clone()});
    }
    if (options_.deskew) {
        localDiagnostics.residualDeskewDegrees = estimateSkewDegrees(output);
        output = deskew(output, localDiagnostics.residualDeskewDegrees);
    }
    if (captureStageImages) {
        localDiagnostics.stages.push_back({"deskewed", output.clone()});
    }
    if (options_.denoise) {
        output = denoise(output);
    }
    if (captureStageImages) {
        localDiagnostics.stages.push_back({"denoised", output.clone()});
    }
    if (options_.normalizeContrast) {
        output = normalizeContrast(output);
    }
    if (captureStageImages) {
        localDiagnostics.stages.push_back({"contrast_normalized", output.clone()});
    }
    if (diagnostics != nullptr) {
        *diagnostics = localDiagnostics;
    }
    return output;
}

double Preprocessor::estimateSkewDegrees(const cv::Mat& input) const {
    if (input.empty()) {
        throw std::invalid_argument("Cannot estimate skew from an empty image");
    }

    cv::Mat gray = toGray(input);
    // Orientation search needs only text-line structure. Run it on a small
    // preview to avoid rotating several megapixels for every candidate angle.
    const double previewScale = std::min(1.0, 900.0 / std::max(gray.cols, gray.rows));
    if (previewScale < 1.0) {
        cv::resize(gray, gray, cv::Size(), previewScale, previewScale, cv::INTER_AREA);
    }
    cv::Mat binary;
    // Keep text as white foreground and suppress isolated noise before
    // comparing horizontal projection concentration at candidate angles.
    cv::threshold(gray, binary, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);
    cv::morphologyEx(binary, binary, cv::MORPH_OPEN,
                     cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2)));
    const double foregroundFraction = cv::countNonZero(binary) /
        (static_cast<double>(binary.rows) * binary.cols);
    if (cv::countNonZero(binary) < 20 || foregroundFraction > 0.45) {
        return 0.0;
    }

    // Search broadly first, then refine around the best angle. This estimates
    // line orientation rather than the bounding box of all page contents, so
    // figures and margins have less influence than in the minAreaRect method.
    const double limit = std::min(options_.maxDeskewDegrees, 30.0);
    double bestAngle = 0.0;
    double bestScore = -1.0;
    cv::Point2f center(binary.cols / 2.0F, binary.rows / 2.0F);
    auto search = [&](double start, double stop, double step) {
        for (double angle = start; angle <= stop + step * 0.25; angle += step) {
            cv::Mat rotation = cv::getRotationMatrix2D(center, angle, 1.0);
            cv::Mat rotated;
            cv::warpAffine(binary, rotated, rotation, binary.size(), cv::INTER_LINEAR,
                           cv::BORDER_CONSTANT, cv::Scalar(0));
            const double score = horizontalProjectionScore(rotated);
            if (score > bestScore) {
                bestScore = score;
                bestAngle = angle;
            }
        }
    };
    search(-limit, limit, 1.0);
    const double coarseBest = bestAngle;
    search(std::max(-limit, coarseBest - 1.1), std::min(limit, coarseBest + 1.1), 0.2);
    return std::abs(bestAngle) < 0.15 ? 0.0 : bestAngle;
}

cv::Mat Preprocessor::rectifyPage(const cv::Mat& input, PreprocessDiagnostics* diagnostics) const {
    cv::Mat gray = toGray(input);

    // Work on a bounded-size copy for corner detection; map the selected
    // quadrilateral back to source coordinates before warping the full image.
    const double scale = std::min(1.0, 1200.0 / std::max(input.cols, input.rows));
    cv::Mat small;
    if (scale < 1.0) {
        cv::resize(gray, small, cv::Size(), scale, scale, cv::INTER_AREA);
    } else {
        small = gray;
    }
    cv::GaussianBlur(small, small, cv::Size(5, 5), 0.0);

    std::vector<std::vector<cv::Point>> contours;
    auto collectContours = [&](cv::Mat mask) {
        std::vector<std::vector<cv::Point>> found;
        cv::findContours(mask, found, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
        contours.insert(contours.end(), found.begin(), found.end());
    };

    cv::Mat edges;
    if (cudaEnabled_) {
        cv::cuda::GpuMat gpuSmall, gpuEdges;
        gpuSmall.upload(small);
        auto canny = cv::cuda::createCannyEdgeDetector(40.0, 130.0);
        canny->detect(gpuSmall, gpuEdges);
        gpuEdges.download(edges);
    } else {
        cv::Canny(small, edges, 40, 130);
    }
    cv::morphologyEx(edges, edges, cv::MORPH_CLOSE,
                     cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5)));
    if (diagnostics != nullptr && diagnostics->captureStageImages) {
        diagnostics->stages.push_back({"page_edges", edges.clone()});
    }
    collectContours(edges);

    // Threshold contours help when the paper has a uniform tone against a
    // contrasting desk and its outline is not a clean Canny contour.
    cv::Mat thresholded;
    cv::threshold(small, thresholded, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    for (int polarity : {0, 1}) {
        cv::Mat mask = polarity == 0 ? thresholded.clone() : 255 - thresholded;
        cv::morphologyEx(mask, mask, cv::MORPH_CLOSE,
                         cv::getStructuringElement(cv::MORPH_RECT, cv::Size(9, 9)));
        collectContours(mask);
    }

    const double imageArea = static_cast<double>(small.rows) * small.cols;
    double bestArea = 0.0;
    std::optional<std::array<cv::Point2f, 4>> bestCorners;
    for (const auto& contour : contours) {
        std::vector<cv::Point> hull;
        cv::convexHull(contour, hull);
        const double perimeter = cv::arcLength(hull, true);
        std::vector<cv::Point> polygon;
        cv::approxPolyDP(hull, polygon, 0.02 * perimeter, true);
        if (polygon.size() != 4 || !cv::isContourConvex(polygon)) {
            continue;
        }
        const double area = std::abs(cv::contourArea(polygon));
        const double areaFraction = area / imageArea;
        // findContours can treat the outer image frame as a huge quadrilateral.
        // It is not a detected sheet; prefer the next-largest plausible object.
        if (areaFraction < 0.15 || areaFraction > 0.97 || area <= bestArea) {
            continue;
        }
        bestArea = area;
        bestCorners = orderCorners(polygon);
    }

    cv::Mat boundaryView;
    cv::cvtColor(small, boundaryView, cv::COLOR_GRAY2BGR);
    if (!bestCorners) {
        if (diagnostics != nullptr && diagnostics->captureStageImages) {
            diagnostics->stages.push_back({"page_boundary_overlay", boundaryView.clone()});
        }
        return input;
    }

    std::array<cv::Point2f, 4> source = *bestCorners;
    std::vector<cv::Point> overlayCorners;
    overlayCorners.reserve(source.size());
    for (const auto& point : source) {
        overlayCorners.emplace_back(cvRound(point.x), cvRound(point.y));
    }
    cv::polylines(boundaryView, overlayCorners, true, cv::Scalar(0, 0, 255), 3, cv::LINE_AA);
    for (const auto& point : overlayCorners) {
        cv::circle(boundaryView, point, 8, cv::Scalar(0, 255, 0), cv::FILLED, cv::LINE_AA);
    }
    if (diagnostics != nullptr && diagnostics->captureStageImages) {
        diagnostics->stages.push_back({"page_boundary_overlay", boundaryView.clone()});
    }
    if (scale < 1.0) {
        for (auto& point : source) {
            point.x = static_cast<float>(point.x / scale);
            point.y = static_cast<float>(point.y / scale);
        }
    }
    if (diagnostics != nullptr) {
        diagnostics->pageBoundaryDetected = true;
        diagnostics->pageCorners.assign(source.begin(), source.end());
    }
    const auto& tl = source[0];
    const auto& tr = source[1];
    const auto& br = source[2];
    const auto& bl = source[3];
    const int width = static_cast<int>(std::round(std::max(cv::norm(tr - tl), cv::norm(br - bl))));
    const int height = static_cast<int>(std::round(std::max(cv::norm(bl - tl), cv::norm(br - tr))));
    if (width < 2 || height < 2) {
        return input;
    }

    const std::array<cv::Point2f, 4> destination = {
        cv::Point2f(0.0F, 0.0F), cv::Point2f(static_cast<float>(width - 1), 0.0F),
        cv::Point2f(static_cast<float>(width - 1), static_cast<float>(height - 1)),
        cv::Point2f(0.0F, static_cast<float>(height - 1))};
    cv::Mat transform = cv::getPerspectiveTransform(source.data(), destination.data());
    cv::Mat output;
    if (cudaEnabled_) {
        cv::cuda::GpuMat gpuInput, gpuOutput;
        gpuInput.upload(input);
        cv::cuda::warpPerspective(gpuInput, gpuOutput, transform, cv::Size(width, height),
                                  cv::INTER_CUBIC, cv::BORDER_REPLICATE);
        gpuOutput.download(output);
    } else {
        cv::warpPerspective(input, output, transform, cv::Size(width, height), cv::INTER_CUBIC,
                            cv::BORDER_REPLICATE);
    }
    return output;
}

cv::Mat Preprocessor::deskew(const cv::Mat& input, double angle) const {
    if (std::abs(angle) < 0.1 || std::abs(angle) > options_.maxDeskewDegrees) {
        return input;
    }

    const cv::Point2f center(input.cols / 2.0F, input.rows / 2.0F);
    cv::Mat rotation = cv::getRotationMatrix2D(center, angle, 1.0);

    // Expand the canvas so rotation never cuts off text near the page edges.
    const double radians = std::abs(angle) * CV_PI / 180.0;
    const int outWidth = static_cast<int>(input.cols * std::cos(radians) + input.rows * std::sin(radians));
    const int outHeight = static_cast<int>(input.rows * std::cos(radians) + input.cols * std::sin(radians));
    rotation.at<double>(0, 2) += outWidth / 2.0 - center.x;
    rotation.at<double>(1, 2) += outHeight / 2.0 - center.y;

    cv::Mat output;
    const cv::Scalar paper = input.channels() == 1 ? cv::Scalar(255) : cv::Scalar(255, 255, 255);
    if (cudaEnabled_) {
        cv::cuda::GpuMat gpuInput, gpuOutput;
        gpuInput.upload(input);
        cv::cuda::warpAffine(gpuInput, gpuOutput, rotation, cv::Size(outWidth, outHeight),
                             cv::INTER_CUBIC, cv::BORDER_CONSTANT, paper);
        gpuOutput.download(output);
    } else {
        cv::warpAffine(input, output, rotation, cv::Size(outWidth, outHeight), cv::INTER_CUBIC,
                       cv::BORDER_CONSTANT, paper);
    }
    return output;
}

cv::Mat Preprocessor::denoise(const cv::Mat& input) const {
    if (options_.denoiseStrength == 0.0F) {
        return input;
    }
    cv::Mat output;
    if (cudaEnabled_) {
        cv::cuda::GpuMat gpuInput, gpuOutput;
        gpuInput.upload(input);
        if (input.channels() == 1) {
            cv::cuda::fastNlMeansDenoising(gpuInput, gpuOutput, options_.denoiseStrength, 21, 7);
        } else {
            cv::cuda::fastNlMeansDenoisingColored(gpuInput, gpuOutput, options_.denoiseStrength,
                                                  options_.denoiseStrength, 21, 7);
        }
        gpuOutput.download(output);
    } else if (input.channels() == 1) {
        cv::fastNlMeansDenoising(input, output, options_.denoiseStrength, 7, 21);
    } else {
        cv::fastNlMeansDenoisingColored(input, output, options_.denoiseStrength,
                                        options_.denoiseStrength, 7, 21);
    }
    return output;
}

cv::Mat Preprocessor::normalizeContrast(const cv::Mat& input) const {
    cv::Mat output;
    if (cudaEnabled_) {
        cv::cuda::GpuMat gpuInput, gpuOutput;
        gpuInput.upload(input);
        auto clahe = cv::cuda::createCLAHE(options_.claheClipLimit, options_.claheTileGrid);
        if (input.channels() == 1) {
            clahe->apply(gpuInput, gpuOutput);
        } else {
            cv::cuda::GpuMat lab;
            cv::cuda::cvtColor(gpuInput, lab, cv::COLOR_BGR2Lab);
            std::vector<cv::cuda::GpuMat> channels;
            cv::cuda::split(lab, channels);
            clahe->apply(channels[0], channels[0]);
            cv::cuda::merge(channels, lab);
            cv::cuda::cvtColor(lab, gpuOutput, cv::COLOR_Lab2BGR);
        }
        gpuOutput.download(output);
        return output;
    }

    auto clahe = cv::createCLAHE(options_.claheClipLimit, options_.claheTileGrid);
    if (input.channels() == 1) {
        clahe->apply(input, output);
        return output;
    }

    // Apply CLAHE to luminance only so the page's original colors are retained.
    cv::Mat lab;
    cv::cvtColor(input, lab, cv::COLOR_BGR2Lab);
    std::vector<cv::Mat> channels;
    cv::split(lab, channels);
    clahe->apply(channels[0], channels[0]);
    cv::merge(channels, lab);
    cv::cvtColor(lab, output, cv::COLOR_Lab2BGR);
    return output;
}

} // namespace exam_hacker
