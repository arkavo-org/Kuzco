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
            name: "KuzcoBridge",
            dependencies: ["llama"],
            path: "Sources/KuzcoBridge",
            publicHeadersPath: "include",
            cSettings: [
                .headerSearchPath("../../Vendors/llama.xcframework/macos-arm64_x86_64/Headers"),
                .headerSearchPath("../../Vendors/llama.xcframework/ios-arm64/Headers")
            ]
        ),
        .target(
            name: "Kuzco",
            dependencies: [
                "llama",
                "KuzcoBridge"
            ],
            path: "Sources/Kuzco",
            linkerSettings: [
                .linkedLibrary("c++")
            ]),
        .testTarget(
            name: "KuzcoTests",
            dependencies: ["Kuzco"]),
    ]
)
