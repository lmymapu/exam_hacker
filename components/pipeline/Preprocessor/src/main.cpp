#include "Preprocessor.h"

#include <opencv2/imgcodecs.hpp>

#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>

namespace {

void logPreprocessDiagnostics(const exam_hacker::PreprocessDiagnostics& diagnostics) {
    std::cout << "[Preprocessor] CUDA acceleration: "
              << (diagnostics.cudaEnabled ? "enabled" : "unavailable; using CPU fallback") << '\n'
              << "[Preprocessor] Paper boundary detected: "
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
    if (argc < 3) {
        std::cerr << "Usage: exam_preprocess <input-image> <output-image> "
                     "[--device cpu|gpu] [--debug-dir <directory>]\n"
                     "       A positional debug directory is also accepted for compatibility.\n";
        return 2;
    }

    try {
        bool useCuda = true;
        std::optional<std::filesystem::path> debugDirectory;
        for (int i = 3; i < argc; ++i) {
            const std::string argument = argv[i];
            if (argument == "--device") {
                if (i + 1 >= argc) {
                    throw std::invalid_argument("--device requires either 'cpu' or 'gpu'");
                }
                const std::string device = argv[++i];
                if (device == "cpu") {
                    useCuda = false;
                } else if (device == "gpu") {
                    useCuda = true;
                } else {
                    throw std::invalid_argument("--device must be either 'cpu' or 'gpu'");
                }
            } else if (argument == "--debug-dir") {
                if (i + 1 >= argc) {
                    throw std::invalid_argument("--debug-dir requires a directory path");
                }
                debugDirectory = argv[++i];
            } else if (!argument.empty() && argument[0] != '-' && !debugDirectory) {
                // Keep accepting the prior third positional debug-directory argument.
                debugDirectory = argument;
            } else {
                throw std::invalid_argument("Unknown or repeated argument: " + argument);
            }
        }

        const cv::Mat input = cv::imread(argv[1], cv::IMREAD_COLOR);
        if (input.empty()) {
            std::cerr << "Could not read input image: " << argv[1] << '\n';
            return 1;
        }

        exam_hacker::PreprocessorOptions options;
        options.useCuda = useCuda;
        exam_hacker::Preprocessor preprocessor(options);
        exam_hacker::PreprocessDiagnostics diagnostics;
        diagnostics.captureStageImages = debugDirectory.has_value();
        const cv::Mat output = preprocessor.process(input, &diagnostics);
        std::cout << "[Preprocessor] Requested backend: " << (useCuda ? "GPU" : "CPU only") << '\n';
        logPreprocessDiagnostics(diagnostics);
        if (!cv::imwrite(argv[2], output)) {
            std::cerr << "Could not write output image: " << argv[2] << '\n';
            return 1;
        }

        std::cout << "Wrote preprocessed image to " << argv[2] << '\n';
        if (debugDirectory) {
            saveDiagnostics(diagnostics, *debugDirectory);
            std::cout << "Wrote preprocessing stages and metadata to " << debugDirectory->string() << '\n';
        }
    } catch (const std::exception& error) {
        std::cerr << "Preprocessing failed: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
