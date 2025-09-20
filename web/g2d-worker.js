// Web Worker to parse .g2d binary files and return typed arrays

/**
Header (little-endian, 32 bytes total):
  - magic: 4 bytes ('G2D\0')
  - version: u32
  - num_gaussians: u32
  - img_width: u32
  - img_height: u32
  - feat_channels: u32
  - quantization_bits: u32
  - flags: u32

Data (float32):
  - xy: [N,2]
  - scale: [N,2]
  - rot: [N,1]
  - feat: [N,C]
*/

self.onmessage = (e) => {
  const { buffer } = e.data;
  try {
    const dv = new DataView(buffer);
    const u8 = new Uint8Array(buffer);

    // Magic
    if (u8[0] !== 0x47 || u8[1] !== 0x32 || u8[2] !== 0x44 || u8[3] !== 0x00) {
      throw new Error('Invalid G2D magic');
    }

    const version = dv.getUint32(4, true);
    const num = dv.getUint32(8, true);
    const imgW = dv.getUint32(12, true);
    const imgH = dv.getUint32(16, true);
    const ch = dv.getUint32(20, true);
    const quantBits = dv.getUint32(24, true);
    const flags = dv.getUint32(28, true);

    const expectedData = num * (2 + 2 + 1 + ch) * 4;
    const expectedTotal = 32 + expectedData;
    if (buffer.byteLength !== expectedTotal) {
      throw new Error(`Invalid size: expected ${expectedTotal}, got ${buffer.byteLength}`);
    }

    let off = 32;
    const xy = new Float32Array(buffer, off, num * 2); off += num * 2 * 4;
    const scale = new Float32Array(buffer, off, num * 2); off += num * 2 * 4;
    const rot = new Float32Array(buffer, off, num * 1); off += num * 1 * 4;
    const feat = new Float32Array(buffer, off, num * ch);

    // Copy into new ArrayBuffers to avoid transferring the entire original buffer
    const xyCopy = new Float32Array(num * 2); xyCopy.set(xy);
    const scaleCopy = new Float32Array(num * 2); scaleCopy.set(scale);
    const rotCopy = new Float32Array(num * 1); rotCopy.set(rot);
    const featCopy = new Float32Array(num * ch); featCopy.set(feat);

    self.postMessage({
      version, num, imgW, imgH, ch, quantBits, flags,
      xy: xyCopy.buffer,
      scale: scaleCopy.buffer,
      rot: rotCopy.buffer,
      feat: featCopy.buffer,
    }, [xyCopy.buffer, scaleCopy.buffer, rotCopy.buffer, featCopy.buffer]);
  } catch (err) {
    self.postMessage({ error: String(err) });
  }
};


