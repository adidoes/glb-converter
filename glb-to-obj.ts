#!/usr/bin/env bun

import {
  existsSync,
  mkdirSync,
  mkdtempSync,
  readdirSync,
  readFileSync,
  rmSync,
  statSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { basename, dirname, extname, join, resolve } from "node:path";
import draco3dImport from "draco3dgltf";

type Gltf = {
  scene?: number;
  scenes?: Array<{ nodes?: number[] }>;
  nodes?: GltfNode[];
  meshes?: GltfMesh[];
  accessors?: GltfAccessor[];
  bufferViews?: GltfBufferView[];
  buffers?: Array<{ uri?: string; byteLength?: number }>;
  materials?: GltfMaterial[];
  textures?: GltfTexture[];
  images?: GltfImage[];
};

type GltfNode = {
  name?: string;
  mesh?: number;
  children?: number[];
  matrix?: number[];
  translation?: number[];
  rotation?: number[];
  scale?: number[];
};

type GltfMesh = {
  name?: string;
  primitives?: GltfPrimitive[];
};

type GltfPrimitive = {
  mode?: number;
  attributes?: Record<string, number>;
  indices?: number;
  material?: number;
  extensions?: {
    KHR_draco_mesh_compression?: DracoMeshCompression;
    [key: string]: unknown;
  };
};

type DracoMeshCompression = {
  bufferView: number;
  attributes: Record<string, number>;
};

type GltfAccessor = {
  bufferView?: number;
  byteOffset?: number;
  componentType: number;
  count: number;
  type: string;
  normalized?: boolean;
  sparse?: unknown;
};

type GltfBufferView = {
  buffer?: number;
  byteOffset?: number;
  byteLength: number;
  byteStride?: number;
};

type GltfMaterial = {
  name?: string;
  alphaMode?: string;
  doubleSided?: boolean;
  emissiveFactor?: number[];
  pbrMetallicRoughness?: {
    baseColorFactor?: number[];
    baseColorTexture?: GltfTextureInfo;
    metallicFactor?: number;
    roughnessFactor?: number;
    metallicRoughnessTexture?: GltfTextureInfo;
  };
  normalTexture?: GltfTextureInfo;
  occlusionTexture?: GltfTextureInfo;
  emissiveTexture?: GltfTextureInfo;
};

type GltfTextureInfo = {
  index: number;
};

type GltfTexture = {
  source?: number;
  extensions?: {
    EXT_texture_webp?: { source?: number };
    [key: string]: unknown;
  };
};

type GltfImage = {
  name?: string;
  uri?: string;
  mimeType?: string;
  bufferView?: number;
};

type Vec2 = [number, number];
type Vec3 = [number, number, number];
type Mat4 = [
  number, number, number, number,
  number, number, number, number,
  number, number, number, number,
  number, number, number, number,
];

const GLB_MAGIC = 0x46546c67;
const CHUNK_JSON = 0x4e4f534a;
const CHUNK_BIN = 0x004e4942;
const MODE_TRIANGLES = 4;

const COMPONENT_SIZES = new Map<number, number>([
  [5120, 1],
  [5121, 1],
  [5122, 2],
  [5123, 2],
  [5125, 4],
  [5126, 4],
]);

const TYPE_COMPONENTS = new Map<string, number>([
  ["SCALAR", 1],
  ["VEC2", 2],
  ["VEC3", 3],
  ["VEC4", 4],
  ["MAT2", 4],
  ["MAT3", 9],
  ["MAT4", 16],
]);

type ObjParts = {
  lines: string[];
  vertexOffset: number;
  uvOffset: number;
  normalOffset: number;
};

type PrimitiveGeometry = {
  positions: Vec3[];
  normals?: Vec3[];
  uvs?: Vec2[];
  indices: number[];
};

type TextureArtifact = {
  relativePath: string;
  bytes: Uint8Array;
};

type ConvertedArtifacts = {
  obj: string;
  mtl?: string;
  textures: TextureArtifact[];
};

type MaterialLibrary = {
  mtl?: string;
  materialNames: Map<number, string>;
  textures: TextureArtifact[];
};

type DracoModule = {
  Decoder: new () => DracoDecoder;
  DecoderBuffer: new () => DracoDecoderBuffer;
  Mesh: new () => DracoMesh;
  DracoFloat32Array: new () => DracoFloat32Array;
  TRIANGULAR_MESH: number;
  HEAPU32: Uint32Array;
  _malloc(size: number): number;
  _free(pointer: number): void;
  destroy(value: object): void;
};

type DracoDecoder = {
  GetEncodedGeometryType(buffer: DracoDecoderBuffer): number;
  GetEncodedGeometryType_Deprecated(buffer: DracoDecoderBuffer): number;
  DecodeBufferToMesh(buffer: DracoDecoderBuffer, mesh: DracoMesh): DracoStatus;
  GetAttributeByUniqueId(mesh: DracoMesh, uniqueId: number): DracoAttribute;
  GetAttributeFloatForAllPoints(mesh: DracoMesh, attribute: DracoAttribute, values: DracoFloat32Array): boolean;
  GetTrianglesUInt32Array(mesh: DracoMesh, byteLength: number, pointer: number): boolean;
};

type DracoDecoderBuffer = {
  Init(data: Uint8Array, byteLength: number): void;
};

type DracoMesh = {
  num_faces(): number;
  num_points(): number;
};

type DracoStatus = {
  ok(): boolean;
  error_msg(): string;
};

type DracoAttribute = {
  num_components(): number;
};

type DracoFloat32Array = {
  GetValue(index: number): number;
  size(): number;
};

type DracoPackage = {
  createDecoderModule(options: Record<string, never>): Promise<DracoModule>;
};

const draco3d = draco3dImport as DracoPackage;
let dracoModulePromise: Promise<DracoModule> | undefined;

function parseGlb(bytes: Uint8Array): { gltf: Gltf; bin: Uint8Array } {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);

  if (view.byteLength < 20 || view.getUint32(0, true) !== GLB_MAGIC) {
    throw new Error("Not a binary .glb file.");
  }

  const version = view.getUint32(4, true);
  if (version !== 2) {
    throw new Error(`Unsupported GLB version ${version}; only GLB 2.0 is supported.`);
  }

  const declaredLength = view.getUint32(8, true);
  if (declaredLength > view.byteLength) {
    throw new Error("GLB file is truncated.");
  }

  let offset = 12;
  let json: Gltf | undefined;
  let bin: Uint8Array | undefined;

  while (offset + 8 <= declaredLength) {
    const chunkLength = view.getUint32(offset, true);
    const chunkType = view.getUint32(offset + 4, true);
    const chunkStart = offset + 8;
    const chunkEnd = chunkStart + chunkLength;

    if (chunkEnd > declaredLength) {
      throw new Error("GLB chunk extends past end of file.");
    }

    const chunk = bytes.subarray(chunkStart, chunkEnd);
    if (chunkType === CHUNK_JSON) {
      json = JSON.parse(new TextDecoder().decode(chunk).trim()) as Gltf;
    } else if (chunkType === CHUNK_BIN) {
      bin = chunk;
    }

    offset = chunkEnd;
  }

  if (!json) {
    throw new Error("GLB is missing its JSON chunk.");
  }
  if (!bin) {
    throw new Error("GLB is missing its binary geometry chunk.");
  }

  return { gltf: json, bin };
}

function getBufferViewBytes(gltf: Gltf, bin: Uint8Array, bufferViewIndex: number): Uint8Array {
  const bufferView = gltf.bufferViews?.[bufferViewIndex];
  if (!bufferView) {
    throw new Error(`Missing bufferView ${bufferViewIndex}.`);
  }
  if (bufferView.buffer !== undefined && bufferView.buffer !== 0) {
    throw new Error("External or multiple GLB buffers are not supported yet.");
  }

  const start = bufferView.byteOffset ?? 0;
  const end = start + bufferView.byteLength;
  if (end > bin.byteLength) {
    throw new Error(`BufferView ${bufferViewIndex} extends past the binary chunk.`);
  }

  return bin.subarray(start, end);
}

async function getDracoModule(): Promise<DracoModule> {
  dracoModulePromise ??= draco3d.createDecoderModule({});
  return dracoModulePromise;
}

function identity(): Mat4 {
  return [
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1,
  ];
}

function multiply(a: Mat4, b: Mat4): Mat4 {
  const out = new Array(16).fill(0) as Mat4;
  for (let column = 0; column < 4; column++) {
    for (let row = 0; row < 4; row++) {
      out[column * 4 + row] =
        a[0 * 4 + row] * b[column * 4 + 0] +
        a[1 * 4 + row] * b[column * 4 + 1] +
        a[2 * 4 + row] * b[column * 4 + 2] +
        a[3 * 4 + row] * b[column * 4 + 3];
    }
  }
  return out;
}

function nodeMatrix(node: GltfNode): Mat4 {
  if (node.matrix) {
    if (node.matrix.length !== 16) {
      throw new Error(`Node "${node.name ?? "(unnamed)"}" has an invalid matrix.`);
    }
    return node.matrix as Mat4;
  }

  const [tx, ty, tz] = node.translation ?? [0, 0, 0];
  const [x, y, z, w] = node.rotation ?? [0, 0, 0, 1];
  const [sx, sy, sz] = node.scale ?? [1, 1, 1];

  const x2 = x + x;
  const y2 = y + y;
  const z2 = z + z;
  const xx = x * x2;
  const xy = x * y2;
  const xz = x * z2;
  const yy = y * y2;
  const yz = y * z2;
  const zz = z * z2;
  const wx = w * x2;
  const wy = w * y2;
  const wz = w * z2;

  return [
    (1 - (yy + zz)) * sx, (xy + wz) * sx, (xz - wy) * sx, 0,
    (xy - wz) * sy, (1 - (xx + zz)) * sy, (yz + wx) * sy, 0,
    (xz + wy) * sz, (yz - wx) * sz, (1 - (xx + yy)) * sz, 0,
    tx, ty, tz, 1,
  ];
}

function transformPoint(matrix: Mat4, point: Vec3): Vec3 {
  const [x, y, z] = point;
  return [
    matrix[0] * x + matrix[4] * y + matrix[8] * z + matrix[12],
    matrix[1] * x + matrix[5] * y + matrix[9] * z + matrix[13],
    matrix[2] * x + matrix[6] * y + matrix[10] * z + matrix[14],
  ];
}

function transformNormal(matrix: Mat4, normal: Vec3): Vec3 {
  const a00 = matrix[0];
  const a01 = matrix[4];
  const a02 = matrix[8];
  const a10 = matrix[1];
  const a11 = matrix[5];
  const a12 = matrix[9];
  const a20 = matrix[2];
  const a21 = matrix[6];
  const a22 = matrix[10];
  const det =
    a00 * (a11 * a22 - a12 * a21) -
    a01 * (a10 * a22 - a12 * a20) +
    a02 * (a10 * a21 - a11 * a20);

  const [x, y, z] = normal;
  let nx: number;
  let ny: number;
  let nz: number;

  if (Math.abs(det) < Number.EPSILON) {
    nx = a00 * x + a01 * y + a02 * z;
    ny = a10 * x + a11 * y + a12 * z;
    nz = a20 * x + a21 * y + a22 * z;
  } else {
    const invDet = 1 / det;
    const inv00 = (a11 * a22 - a12 * a21) * invDet;
    const inv01 = (a02 * a21 - a01 * a22) * invDet;
    const inv02 = (a01 * a12 - a02 * a11) * invDet;
    const inv10 = (a12 * a20 - a10 * a22) * invDet;
    const inv11 = (a00 * a22 - a02 * a20) * invDet;
    const inv12 = (a02 * a10 - a00 * a12) * invDet;
    const inv20 = (a10 * a21 - a11 * a20) * invDet;
    const inv21 = (a01 * a20 - a00 * a21) * invDet;
    const inv22 = (a00 * a11 - a01 * a10) * invDet;

    nx = inv00 * x + inv10 * y + inv20 * z;
    ny = inv01 * x + inv11 * y + inv21 * z;
    nz = inv02 * x + inv12 * y + inv22 * z;
  }

  const length = Math.hypot(nx, ny, nz) || 1;
  return [nx / length, ny / length, nz / length];
}

function readComponent(view: DataView, offset: number, componentType: number, normalized = false): number {
  switch (componentType) {
    case 5120: {
      const value = view.getInt8(offset);
      return normalized ? Math.max(value / 127, -1) : value;
    }
    case 5121: {
      const value = view.getUint8(offset);
      return normalized ? value / 255 : value;
    }
    case 5122: {
      const value = view.getInt16(offset, true);
      return normalized ? Math.max(value / 32767, -1) : value;
    }
    case 5123: {
      const value = view.getUint16(offset, true);
      return normalized ? value / 65535 : value;
    }
    case 5125:
      return view.getUint32(offset, true);
    case 5126:
      return view.getFloat32(offset, true);
    default:
      throw new Error(`Unsupported component type ${componentType}.`);
  }
}

function readAccessor(gltf: Gltf, bin: Uint8Array, accessorIndex: number): number[][] {
  const accessor = gltf.accessors?.[accessorIndex];
  if (!accessor) {
    throw new Error(`Missing accessor ${accessorIndex}.`);
  }
  if (accessor.sparse) {
    throw new Error(`Sparse accessor ${accessorIndex} is not supported yet.`);
  }
  if (accessor.bufferView === undefined) {
    throw new Error(`Accessor ${accessorIndex} has no bufferView.`);
  }

  const bufferView = gltf.bufferViews?.[accessor.bufferView];
  if (!bufferView) {
    throw new Error(`Missing bufferView ${accessor.bufferView}.`);
  }
  if (bufferView.buffer !== undefined && bufferView.buffer !== 0) {
    throw new Error("External or multiple GLB buffers are not supported yet.");
  }

  const componentSize = COMPONENT_SIZES.get(accessor.componentType);
  const componentCount = TYPE_COMPONENTS.get(accessor.type);
  if (!componentSize || !componentCount) {
    throw new Error(`Unsupported accessor format ${accessor.type}/${accessor.componentType}.`);
  }

  const stride = bufferView.byteStride ?? componentSize * componentCount;
  const start = (bufferView.byteOffset ?? 0) + (accessor.byteOffset ?? 0);
  const end = (bufferView.byteOffset ?? 0) + bufferView.byteLength;
  const view = new DataView(bin.buffer, bin.byteOffset, bin.byteLength);
  const values: number[][] = [];

  for (let row = 0; row < accessor.count; row++) {
    const rowOffset = start + row * stride;
    if (rowOffset + componentSize * componentCount > end) {
      throw new Error(`Accessor ${accessorIndex} reads past its bufferView.`);
    }

    const tuple: number[] = [];
    for (let component = 0; component < componentCount; component++) {
      tuple.push(readComponent(
        view,
        rowOffset + component * componentSize,
        accessor.componentType,
        accessor.normalized,
      ));
    }
    values.push(tuple);
  }

  return values;
}

function sanitizeName(name: string): string {
  return name.trim().replace(/\s+/g, "_").replace(/[^A-Za-z0-9_.-]/g, "_") || "mesh";
}

function formatNumber(value: number): string {
  if (Object.is(value, -0)) {
    return "0";
  }
  return Number.isInteger(value) ? String(value) : Number(value.toFixed(8)).toString();
}

function imageExtension(mimeType?: string, uri?: string): string {
  if (mimeType === "image/webp" || uri?.split("?")[0].toLowerCase().endsWith(".webp")) {
    return ".png";
  }
  if (mimeType === "image/jpeg") {
    return ".jpg";
  }
  if (mimeType === "image/png") {
    return ".png";
  }
  const uriExt = uri ? extname(uri.split("?")[0]) : "";
  return uriExt || ".bin";
}

function isWebpImage(image: GltfImage): boolean {
  return image.mimeType === "image/webp" || Boolean(image.uri?.split("?")[0].toLowerCase().endsWith(".webp"));
}

function convertWebpToPng(bytes: Uint8Array): Uint8Array {
  const workdir = mkdtempSync(join(tmpdir(), "glb-to-obj-webp-"));
  const input = join(workdir, "input.webp");
  const output = join(workdir, "output.png");

  try {
    writeFileSync(input, bytes);
    const result = Bun.spawnSync({
      cmd: ["sips", "-s", "format", "png", input, "--out", output],
      stdout: "ignore",
      stderr: "pipe",
    });

    if (!result.success) {
      const stderr = new TextDecoder().decode(result.stderr).trim();
      throw new Error(`WebP to PNG conversion failed${stderr ? `: ${stderr}` : ""}`);
    }

    return readFileSync(output);
  } finally {
    rmSync(workdir, { recursive: true, force: true });
  }
}

function decodeDataUri(uri: string): Uint8Array | undefined {
  const match = /^data:([^;,]+)?(;base64)?,(.*)$/i.exec(uri);
  if (!match) {
    return undefined;
  }

  const payload = decodeURIComponent(match[3]);
  if (match[2]) {
    return new Uint8Array(Buffer.from(payload, "base64"));
  }
  return new TextEncoder().encode(payload);
}

function resolveTextureImageIndex(gltf: Gltf, textureIndex: number): number | undefined {
  const texture = gltf.textures?.[textureIndex];
  return texture?.extensions?.EXT_texture_webp?.source ?? texture?.source;
}

function textureRelativePath(gltf: Gltf, textureDir: string, imageIndex: number): string {
  const image = gltf.images?.[imageIndex];
  const extension = imageExtension(image?.mimeType, image?.uri);
  const name = sanitizeName(image?.name ?? `image_${imageIndex}`);
  return `${textureDir}/${String(imageIndex).padStart(2, "0")}_${name}${extension}`;
}

function textureReferencePath(gltf: Gltf, textureDir: string, imageIndex: number): string | undefined {
  const image = gltf.images?.[imageIndex];
  if (!image) {
    return undefined;
  }

  if (image.bufferView !== undefined || image.uri?.startsWith("data:")) {
    return textureRelativePath(gltf, textureDir, imageIndex);
  }
  return image.uri;
}

function texturePathForMaterial(
  gltf: Gltf,
  material: GltfMaterial,
  slot: keyof Pick<GltfMaterial, "normalTexture" | "occlusionTexture" | "emissiveTexture">,
  textureDir: string,
): string | undefined {
  const textureInfo = material[slot];
  if (!textureInfo) {
    return undefined;
  }

  const imageIndex = resolveTextureImageIndex(gltf, textureInfo.index);
  return imageIndex === undefined ? undefined : textureReferencePath(gltf, textureDir, imageIndex);
}

function pbrTexturePath(
  gltf: Gltf,
  textureInfo: GltfTextureInfo | undefined,
  textureDir: string,
): string | undefined {
  if (!textureInfo) {
    return undefined;
  }

  const imageIndex = resolveTextureImageIndex(gltf, textureInfo.index);
  return imageIndex === undefined ? undefined : textureReferencePath(gltf, textureDir, imageIndex);
}

function extractTextureArtifacts(gltf: Gltf, bin: Uint8Array, textureDir: string): TextureArtifact[] {
  const referencedImages = new Set<number>();

  for (const material of gltf.materials ?? []) {
    const pbr = material.pbrMetallicRoughness;
    for (const textureInfo of [
      pbr?.baseColorTexture,
      pbr?.metallicRoughnessTexture,
      material.normalTexture,
      material.occlusionTexture,
      material.emissiveTexture,
    ]) {
      if (!textureInfo) {
        continue;
      }
      const imageIndex = resolveTextureImageIndex(gltf, textureInfo.index);
      if (imageIndex !== undefined) {
        referencedImages.add(imageIndex);
      }
    }
  }

  const artifacts: TextureArtifact[] = [];
  for (const imageIndex of referencedImages) {
    const image = gltf.images?.[imageIndex];
    if (!image) {
      continue;
    }

    let bytes: Uint8Array | undefined;
    if (image.bufferView !== undefined) {
      bytes = getBufferViewBytes(gltf, bin, image.bufferView);
    } else if (image.uri?.startsWith("data:")) {
      bytes = decodeDataUri(image.uri);
    }

    if (bytes) {
      if (isWebpImage(image)) {
        bytes = convertWebpToPng(bytes);
      }

      artifacts.push({
        relativePath: textureRelativePath(gltf, textureDir, imageIndex),
        bytes,
      });
    }
  }

  return artifacts;
}

function buildMaterialLibrary(gltf: Gltf, bin: Uint8Array, sourceBase: string): MaterialLibrary {
  const materials = gltf.materials ?? [];
  if (materials.length === 0) {
    return { materialNames: new Map(), textures: [] };
  }

  const textureDir = `${sanitizeName(sourceBase)}_textures`;
  const textures = extractTextureArtifacts(gltf, bin, textureDir);
  const materialNames = new Map<number, string>();
  const usedNames = new Set<string>();
  const lines = [
    `# Converted from ${sourceBase}.glb by glb-to-obj.ts`,
    "# OBJ/MTL supports only a subset of glTF PBR materials.",
  ];

  materials.forEach((material, index) => {
    let materialName = sanitizeName(material.name ?? `material_${index}`);
    while (usedNames.has(materialName)) {
      materialName = `${materialName}_${index}`;
    }
    usedNames.add(materialName);
    materialNames.set(index, materialName);

    const pbr = material.pbrMetallicRoughness;
    const baseColor = pbr?.baseColorFactor ?? [1, 1, 1, 1];
    const emissive = material.emissiveFactor ?? [0, 0, 0];
    const baseColorTexture = pbrTexturePath(gltf, pbr?.baseColorTexture, textureDir);
    const normalTexture = texturePathForMaterial(gltf, material, "normalTexture", textureDir);
    const metallicRoughnessTexture = pbrTexturePath(gltf, pbr?.metallicRoughnessTexture, textureDir);
    const occlusionTexture = texturePathForMaterial(gltf, material, "occlusionTexture", textureDir);
    const emissiveTexture = texturePathForMaterial(gltf, material, "emissiveTexture", textureDir);

    lines.push(
      "",
      `newmtl ${materialName}`,
      `Ka ${formatNumber(emissive[0] ?? 0)} ${formatNumber(emissive[1] ?? 0)} ${formatNumber(emissive[2] ?? 0)}`,
      `Kd ${formatNumber(baseColor[0] ?? 1)} ${formatNumber(baseColor[1] ?? 1)} ${formatNumber(baseColor[2] ?? 1)}`,
      `d ${formatNumber(baseColor[3] ?? 1)}`,
      `illum ${baseColorTexture || normalTexture ? 2 : 1}`,
    );

    if (baseColorTexture) {
      lines.push(`map_Kd ${baseColorTexture}`);
    }
    if (normalTexture) {
      lines.push(`norm ${normalTexture}`);
      lines.push(`bump ${normalTexture}`);
    }
    if (emissiveTexture) {
      lines.push(`map_Ke ${emissiveTexture}`);
    }
    if (metallicRoughnessTexture) {
      lines.push(`# glTF metallic-roughness texture: ${metallicRoughnessTexture}`);
    }
    if (occlusionTexture) {
      lines.push(`# glTF occlusion texture: ${occlusionTexture}`);
    }
    if (material.alphaMode && material.alphaMode !== "OPAQUE") {
      lines.push(`# glTF alpha mode: ${material.alphaMode}`);
    }
    if (material.doubleSided) {
      lines.push("# glTF double-sided material");
    }
  });

  return {
    mtl: `${lines.join("\n")}\n`,
    materialNames,
    textures,
  };
}

async function decodeDracoPrimitive(
  gltf: Gltf,
  bin: Uint8Array,
  primitive: GltfPrimitive,
  name: string,
): Promise<PrimitiveGeometry> {
  const extension = primitive.extensions?.KHR_draco_mesh_compression;
  if (!extension) {
    throw new Error(`${name} is missing its Draco extension data.`);
  }

  const draco = await getDracoModule();
  const data = getBufferViewBytes(gltf, bin, extension.bufferView);
  const decoder = new draco.Decoder();
  const buffer = new draco.DecoderBuffer();
  const mesh = new draco.Mesh();

  try {
    buffer.Init(data, data.byteLength);

    const geometryType = "GetEncodedGeometryType" in decoder
      ? decoder.GetEncodedGeometryType(buffer)
      : decoder.GetEncodedGeometryType_Deprecated(buffer);
    if (geometryType !== draco.TRIANGULAR_MESH) {
      throw new Error(`${name} is not a Draco triangular mesh.`);
    }

    const status = decoder.DecodeBufferToMesh(buffer, mesh);
    if (!status.ok()) {
      throw new Error(`${name} failed Draco decode: ${status.error_msg()}`);
    }

    const decodeFloatAttribute = (semantic: string): number[][] | undefined => {
      const uniqueId = extension.attributes[semantic];
      if (uniqueId === undefined) {
        return undefined;
      }

      const attribute = decoder.GetAttributeByUniqueId(mesh, uniqueId);
      const componentCount = attribute.num_components();
      const values = new draco.DracoFloat32Array();
      try {
        const ok = decoder.GetAttributeFloatForAllPoints(mesh, attribute, values);
        if (!ok) {
          throw new Error(`${name} failed to decode Draco ${semantic} attribute.`);
        }

        const rows: number[][] = [];
        for (let point = 0; point < mesh.num_points(); point++) {
          const row: number[] = [];
          for (let component = 0; component < componentCount; component++) {
            row.push(values.GetValue(point * componentCount + component));
          }
          rows.push(row);
        }
        return rows;
      } finally {
        draco.destroy(values);
      }
    };

    const positions = decodeFloatAttribute("POSITION") as Vec3[] | undefined;
    if (!positions) {
      throw new Error(`${name} has no Draco POSITION attribute.`);
    }

    const normals = decodeFloatAttribute("NORMAL") as Vec3[] | undefined;
    const uvs = decodeFloatAttribute("TEXCOORD_0") as Vec2[] | undefined;
    const indexCount = mesh.num_faces() * 3;
    const indexBytes = indexCount * 4;
    const indexPointer = draco._malloc(indexBytes);
    try {
      const ok = decoder.GetTrianglesUInt32Array(mesh, indexBytes, indexPointer);
      if (!ok) {
        throw new Error(`${name} failed to decode Draco triangle indices.`);
      }
      const indices = Array.from(draco.HEAPU32.subarray(indexPointer / 4, indexPointer / 4 + indexCount));
      return { positions, normals, uvs, indices };
    } finally {
      draco._free(indexPointer);
    }
  } finally {
    draco.destroy(mesh);
    draco.destroy(buffer);
    draco.destroy(decoder);
  }
}

async function readPrimitiveGeometry(
  gltf: Gltf,
  bin: Uint8Array,
  primitive: GltfPrimitive,
  name: string,
): Promise<PrimitiveGeometry> {
  if (primitive.extensions?.KHR_draco_mesh_compression) {
    return decodeDracoPrimitive(gltf, bin, primitive, name);
  }

  const positionAccessor = primitive.attributes?.POSITION;
  if (positionAccessor === undefined) {
    return { positions: [], indices: [] };
  }

  const positions = readAccessor(gltf, bin, positionAccessor) as Vec3[];
  const normals = primitive.attributes?.NORMAL === undefined
    ? undefined
    : readAccessor(gltf, bin, primitive.attributes.NORMAL) as Vec3[];
  const uvs = primitive.attributes?.TEXCOORD_0 === undefined
    ? undefined
    : readAccessor(gltf, bin, primitive.attributes.TEXCOORD_0) as Vec2[];

  const indices = primitive.indices === undefined
    ? positions.map((_, index) => index)
    : readAccessor(gltf, bin, primitive.indices).map(([index]) => index);

  return { positions, normals, uvs, indices };
}

async function convertPrimitive(
  gltf: Gltf,
  bin: Uint8Array,
  primitive: GltfPrimitive,
  matrix: Mat4,
  name: string,
  obj: ObjParts,
  materialName?: string,
): Promise<void> {
  const mode = primitive.mode ?? MODE_TRIANGLES;
  if (mode !== MODE_TRIANGLES) {
    throw new Error(`${name} uses primitive mode ${mode}; only TRIANGLES are supported.`);
  }

  const { positions, normals, uvs, indices } = await readPrimitiveGeometry(gltf, bin, primitive, name);
  if (positions.length === 0) {
    return;
  }

  if (indices.length % 3 !== 0) {
    throw new Error(`${name} does not contain a whole number of triangles.`);
  }

  obj.lines.push(`o ${sanitizeName(name)}`);
  if (materialName) {
    obj.lines.push(`usemtl ${materialName}`);
  }

  for (const position of positions) {
    const [x, y, z] = transformPoint(matrix, position);
    obj.lines.push(`v ${formatNumber(x)} ${formatNumber(y)} ${formatNumber(z)}`);
  }

  if (uvs) {
    for (const [u, v] of uvs) {
      obj.lines.push(`vt ${formatNumber(u)} ${formatNumber(v)}`);
    }
  }

  if (normals) {
    for (const normal of normals) {
      const [x, y, z] = transformNormal(matrix, normal);
      obj.lines.push(`vn ${formatNumber(x)} ${formatNumber(y)} ${formatNumber(z)}`);
    }
  }

  const vertexRef = (index: number): string => {
    const v = obj.vertexOffset + index + 1;
    const vt = uvs ? obj.uvOffset + index + 1 : "";
    const vn = normals ? obj.normalOffset + index + 1 : "";

    if (uvs && normals) {
      return `${v}/${vt}/${vn}`;
    }
    if (uvs) {
      return `${v}/${vt}`;
    }
    if (normals) {
      return `${v}//${vn}`;
    }
    return String(v);
  };

  for (let i = 0; i < indices.length; i += 3) {
    obj.lines.push(`f ${vertexRef(indices[i])} ${vertexRef(indices[i + 1])} ${vertexRef(indices[i + 2])}`);
  }

  obj.vertexOffset += positions.length;
  obj.uvOffset += uvs?.length ?? 0;
  obj.normalOffset += normals?.length ?? 0;
}

async function convertGlbToObj(source: string): Promise<ConvertedArtifacts> {
  const bytes = readFileSync(source);
  const { gltf, bin } = parseGlb(bytes);
  const sceneIndex = gltf.scene ?? 0;
  const scene = gltf.scenes?.[sceneIndex];
  if (!scene) {
    throw new Error(`Missing scene ${sceneIndex}.`);
  }

  if (gltf.buffers?.[0]?.uri) {
    throw new Error("GLB files with external buffer URIs are not supported by this simple converter.");
  }

  const sourceBase = basename(source, extname(source));
  const materialLibrary = buildMaterialLibrary(gltf, bin, sourceBase);
  const headerLines = [
    `# Converted from ${basename(source)} by glb-to-obj.ts`,
    "# Materials are exported to MTL; skins, animations, and cameras are not exported.",
  ];
  if (materialLibrary.mtl) {
    headerLines.push(`mtllib ${sourceBase}.mtl`);
  }

  const obj: ObjParts = {
    lines: headerLines,
    vertexOffset: 0,
    uvOffset: 0,
    normalOffset: 0,
  };

  const visitNode = async (nodeIndex: number, parentMatrix: Mat4, path: string): Promise<void> => {
    const node = gltf.nodes?.[nodeIndex];
    if (!node) {
      throw new Error(`Missing node ${nodeIndex}.`);
    }

    const nodeName = node.name ?? `node_${nodeIndex}`;
    const worldMatrix = multiply(parentMatrix, nodeMatrix(node));
    const nodePath = path ? `${path}_${nodeName}` : nodeName;

    if (node.mesh !== undefined) {
      const mesh = gltf.meshes?.[node.mesh];
      if (!mesh) {
        throw new Error(`Missing mesh ${node.mesh}.`);
      }

      for (const [primitiveIndex, primitive] of (mesh.primitives ?? []).entries()) {
        const meshName = mesh.name ?? `mesh_${node.mesh}`;
        const materialName = primitive.material === undefined
          ? undefined
          : materialLibrary.materialNames.get(primitive.material);
        await convertPrimitive(
          gltf,
          bin,
          primitive,
          worldMatrix,
          `${nodePath}_${meshName}_${primitiveIndex}`,
          obj,
          materialName,
        );
      }
    }

    for (const childIndex of node.children ?? []) {
      await visitNode(childIndex, worldMatrix, nodePath);
    }
  };

  for (const nodeIndex of scene.nodes ?? []) {
    await visitNode(nodeIndex, identity(), "");
  }

  if (obj.vertexOffset === 0) {
    throw new Error("No mesh geometry was found.");
  }

  return {
    obj: `${obj.lines.join("\n")}\n`,
    mtl: materialLibrary.mtl,
    textures: materialLibrary.textures,
  };
}

function isGlbFile(path: string): boolean {
  return statSync(path).isFile() && extname(path).toLowerCase() === ".glb";
}

function collectGlbFiles(paths: string[]): string[] {
  const files: string[] = [];

  for (const path of paths) {
    const resolved = resolve(path);
    if (!existsSync(resolved)) {
      console.warn(`warn: ${path} does not exist; skipping`);
      continue;
    }

    const stats = statSync(resolved);
    if (stats.isDirectory()) {
      for (const entry of readdirSync(resolved)) {
        const candidate = join(resolved, entry);
        if (isGlbFile(candidate)) {
          files.push(candidate);
        }
      }
    } else if (isGlbFile(resolved)) {
      files.push(resolved);
    } else {
      console.warn(`warn: ${path} is not a .glb file or directory; skipping`);
    }
  }

  return [...new Set(files)].sort((a, b) => a.localeCompare(b));
}

function outputPathFor(source: string): string {
  return join(dirname(source), `${basename(source, extname(source))}.obj`);
}

function materialPathFor(source: string): string {
  return join(dirname(source), `${basename(source, extname(source))}.mtl`);
}

function printHelp(): void {
  console.log(`Usage:
  bun run glb-to-obj.ts [--force] [file-or-folder ...]

Defaults:
  With no paths, converts .glb files in the current folder.
  Existing .obj files are skipped unless --force is set.

Examples:
  bun run glb-to-obj.ts
  bun run glb-to-obj.ts model.glb
  bun build --compile ./glb-to-obj.ts --outfile glb-to-obj`);
}

async function main(): Promise<void> {
  const args = process.argv.slice(2);
  const force = args.includes("--force");
  const help = args.includes("--help") || args.includes("-h");
  const paths = args.filter((arg) => arg !== "--force" && arg !== "--help" && arg !== "-h");

  if (help) {
    printHelp();
    return;
  }

  const files = collectGlbFiles(paths.length > 0 ? paths : [process.cwd()]);
  if (files.length === 0) {
    console.log("No .glb files found.");
    return;
  }

  let converted = 0;
  let skipped = 0;
  let failed = 0;

  for (const source of files) {
    const target = outputPathFor(source);
    if (!force && existsSync(target)) {
      console.log(`skip: ${source} -> ${target} already exists`);
      skipped++;
      continue;
    }

    try {
      const artifacts = await convertGlbToObj(source);
      writeFileSync(target, artifacts.obj);
      if (artifacts.mtl) {
        writeFileSync(materialPathFor(source), artifacts.mtl);
      }
      for (const texture of artifacts.textures) {
        const texturePath = join(dirname(source), texture.relativePath);
        mkdirSync(dirname(texturePath), { recursive: true });
        writeFileSync(texturePath, texture.bytes);
      }
      console.log(`ok: ${source} -> ${target}`);
      converted++;
    } catch (error) {
      failed++;
      const message = error instanceof Error ? error.message : String(error);
      console.error(`fail: ${source}: ${message}`);
    }
  }

  console.log(`Done. converted=${converted} skipped=${skipped} failed=${failed}`);
  if (failed > 0) {
    process.exitCode = 1;
  }
}

main().catch((error) => {
  const message = error instanceof Error ? error.message : String(error);
  console.error(`fatal: ${message}`);
  process.exitCode = 1;
});
