#include "Preprocessor.h"

#include <opencv2/imgcodecs.hpp>

#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void logPreprocessDiagnostics(const exam_hacker::PreprocessDiagnostics& diagnostics) {
    std::cout << "[Preprocessor] Paper boundary detected: "
              << (diagnostics.pageBoundaryDetected ? "yes" : "no") << '\n'
              << "[Preprocessor] Estimated residual deskew angle: "
              << std::fixed << std::setprecision(2)
              << diagnostics.residualDeskewDegrees << " degrees\n";
}

void saveDiagnostics(const exam_hacker::PreprocessDiagnostics& diagnostics,
                     const std::filesystem::path& directory) {
    std::filesystem::create_directories(directory);
    std::ofstream metadata(directory / "metadata.json");
    if (!metadata) {
        throw std::runtime_error("Could not create diagnostics metadata in " + directory.string());
    }

    metadata << "{\n  \"page_boundary_detected\": "
             << (diagnostics.pageBoundaryDetected ? "true" : "false")
             << ",\n  \"residual_deskew_degrees\": "
             << std::fixed << std::setprecision(3) << diagnostics.residualDeskewDegrees
             << ",\n  \"page_corners_source_pixels\": [";
    for (std::size_t i = 0; i < diagnostics.pageCorners.size(); ++i) {
        const auto& point = diagnostics.pageCorners[i];
        metadata << (i == 0 ? "" : ", ") << "[" << point.x << ", " << point.y << "]";
    }
    metadata << "],\n  \"stages\": [\n";

    for (std::size_t i = 0; i < diagnostics.stages.size(); ++i) {
        const auto& stage = diagnostics.stages[i];
        const std::string filename = std::to_string(i + 1) + "_" + stage.name + ".png";
        if (!cv::imwrite((directory / filename).string(), stage.image)) {
            throw std::runtime_error("Could not save diagnostics image " + (directory / filename).string());
        }
        metadata << "    {\"name\": \"" << stage.name << "\", \"file\": \"" << filename
                 << "\", \"width\": " << stage.image.cols << ", \"height\": " << stage.image.rows
                 << "}" << (i + 1 == diagnostics.stages.size() ? "\n" : ",\n");
    }
    metadata << "  ]\n}\n";
}

} // namespace

int main(int argc, char* argv[]) {
    if (argc != 3 && argc != 4) {
        std::cerr << "Usage: exam_preprocess <input-image> <output-image> [debug-directory]\n";
        return 2;
    }

    try {
        const cv::Mat input = cv::imread(argv[1], cv::IMREAD_COLOR);
        if (input.empty()) {
            std::cerr << "Could not read input image: " << argv[1] << '\n';
            return 1;
        }

        exam_hacker::Preprocessor preprocessor;
        exam_hacker::PreprocessDiagnostics diagnostics;
        const cv::Mat output = preprocessor.process(input, &diagnostics);
        logPreprocessDiagnostics(diagnostics);
        if (!cv::imwrite(argv[2], output)) {
            std::cerr << "Could not write output image: " << argv[2] << '\n';
            return 1;
        }

        const std::filesystem::path outputPath(argv[2]);
        const std::filesystem::path debugDirectory = argc == 4
            ? std::filesystem::path(argv[3])
            : outputPath.parent_path() / (outputPath.stem().string() + "_debug");
        saveDiagnostics(diagnostics, debugDirectory);

        std::cout << "Wrote preprocessed image to " << argv[2] << '\n';
        std::cout << "Wrote preprocessing stages and metadata to " << debugDirectory.string() << '\n';
    } catch (const std::exception& error) {
        std::cerr << "Preprocessing failed: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
