// Apple Vision face rectangles for every image in a directory.
// usage: bin/vision_faces <image_dir> <out.json>   (compiled by setup.sh)
// Coordinates are source pixels, top-left origin (raw pixel grid, no EXIF rotation applied).
import Foundation
import ImageIO
import Vision

struct Face: Codable {
    let x: Double, y: Double, w: Double, h: Double
    let conf: Double
    let roll: Double?, yaw: Double?, pitch: Double?
}

struct Entry: Codable {
    let file: String
    let w: Int, h: Int
    let faces: [Face]  // largest first
    let error: String?
}

let args = CommandLine.arguments
guard args.count == 3 else {
    FileHandle.standardError.write("usage: vision_faces <image_dir> <out.json>\n".data(using: .utf8)!)
    exit(2)
}
let dir = args[1], outPath = args[2]
let files = ((try? FileManager.default.contentsOfDirectory(atPath: dir)) ?? [])
    .filter { !$0.hasPrefix(".") }
    .sorted()

var entries: [Entry] = []
for f in files {
    autoreleasepool {
        let url = URL(fileURLWithPath: dir).appendingPathComponent(f)
        guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
              let img = CGImageSourceCreateImageAtIndex(src, 0, nil)
        else {
            entries.append(Entry(file: f, w: 0, h: 0, faces: [], error: "decode failed"))
            return
        }
        let W = Double(img.width), H = Double(img.height)
        let req = VNDetectFaceRectanglesRequest()
        do {
            try VNImageRequestHandler(cgImage: img, options: [:]).perform([req])
        } catch {
            entries.append(Entry(file: f, w: img.width, h: img.height, faces: [], error: "\(error)"))
            return
        }
        let faces = (req.results ?? []).map { o -> Face in
            let b = o.boundingBox  // normalized, bottom-left origin
            return Face(x: b.minX * W, y: (1 - b.maxY) * H, w: b.width * W, h: b.height * H,
                        conf: Double(o.confidence),
                        roll: o.roll?.doubleValue, yaw: o.yaw?.doubleValue, pitch: o.pitch?.doubleValue)
        }.sorted { $0.w * $0.h > $1.w * $1.h }
        entries.append(Entry(file: f, w: img.width, h: img.height, faces: faces, error: nil))
    }
}

let enc = JSONEncoder()
enc.outputFormatting = [.prettyPrinted, .sortedKeys]
try enc.encode(entries).write(to: URL(fileURLWithPath: outPath))
let noFace = entries.filter { $0.faces.isEmpty }.count
let errors = entries.filter { $0.error != nil }.count
print("vision: \(entries.count) images, \(noFace) without a face, \(errors) errors -> \(outPath)")
