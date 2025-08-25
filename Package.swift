// swift-tools-version:5.9
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "Kuzco",
    platforms: [
        .macOS(.v12),
        .iOS(.v15),
        .macCatalyst(.v15)
    ],
    products: [
        .library(
            name: "Kuzco",
            targets: ["Kuzco"]),
    ],
    dependencies: [
        // Dependencies removed - using vendored llama.xcframework
    ],
    targets: [
        .binaryTarget(
            name: "llama",
            path: "Vendors/llama.xcframework"
        ),
        .target(
            name: "Kuzco",
            dependencies: [
                "llama"
            ],
            publicHeadersPath: "System/include",
            cSettings: [
                .headerSearchPath("../../Vendors/llama.xcframework/macos-arm64_x86_64/Headers"),
                .headerSearchPath("../../Vendors/llama.xcframework/ios-arm64/Headers")
            ],
            linkerSettings: [
                .linkedLibrary("c++")
            ]),
        .testTarget(
            name: "KuzcoTests",
            dependencies: ["Kuzco"]),
    ]
)
