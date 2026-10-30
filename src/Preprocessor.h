#pragma once

#include <opencv2/core.hpp>

#include <string>
#include <vector>

namespace exam_hacker {

struct PreprocessStageArtifact {
    std::string name;
    cv::Mat image;
};

struct PreprocessDiagnostics {
    bool pageBoundaryDetected = false;
    double residualDeskewDegrees = 0.0;
    std::vector<cv::Point2f> pageCorners;
    std::vector<PreprocessStageArtifact> stages;
};

/** Options for the scan/photo cleanup performed before layout analysis. */
struct PreprocessorOptions {
    bool rectifyPage = true;
    bool deskew = true;
    bool denoise = true;
    bool normalizeContrast = true;

    // Maximum correction in either direction. Larger estimates are treated as
    // unreliable (for example, when the page contains a large diagram).
    double maxDeskewDegrees = 30.0;

    // Fast non-local means strength. Higher values remove more noise but may
    // soften fine printed strokes.
    float denoiseStrength = 5.0F;

    // CLAHE settings used for local contrast normalization.
    double claheClipLimit = 2.0;
    cv::Size claheTileGrid = cv::Size(8, 8);
};

/**
 * OpenCV preprocessing for exam sheet images.
 *
 * Input may be grayscale or 3-channel BGR. Output keeps the same channel
 * count; deskew may expand the spatial dimensions to avoid clipping.
 */
class Preprocessor {
public:
    explicit Preprocessor(PreprocessorOptions options = {});

    cv::Mat process(const cv::Mat& input, PreprocessDiagnostics* diagnostics = nullptr) const;

    // Estimate the correction angle in degrees that process() will apply.
    double estimateSkewDegrees(const cv::Mat& input) const;

private:
    cv::Mat rectifyPage(const cv::Mat& input, PreprocessDiagnostics* diagnostics) const;
    cv::Mat deskew(const cv::Mat& input, double angle) const;
    cv::Mat denoise(const cv::Mat& input) const;
    cv::Mat normalizeContrast(const cv::Mat& input) const;

    PreprocessorOptions options_;
};

} // namespace exam_hacker
