// Minimal WebGPU pipeline that reproduces CUDA kernels in WGSL

async function loadText(path) {
    const sep = path.includes('?') ? '&' : '?';
    const res = await fetch(`${path}${sep}v=${Date.now()}`);
    if (!res.ok) throw new Error(`Failed to fetch ${path}`);
    return await res.text();
}

async function ensureWebGPU() {
    if (!('gpu' in navigator)) {
        const isSecure = window.isSecureContext;
        const host = location.hostname;
        const onLocalhost = host === 'localhost' || host === '127.0.0.1' || host === '::1';
        const proto = location.protocol;
        let hint = '';
        if (!isSecure) {
            hint = `This page is not in a secure context (protocol=${proto}). Use https or http://localhost.`;
        } else if (!onLocalhost && proto !== 'https:') {
            hint = 'Non-local HTTP origins are not secure. Use https or access via localhost.';
        } else {
            hint = 'Your browser may need WebGPU enabled (try Chrome 128+ or enable "Unsafe WebGPU").';
        }
        throw new Error('WebGPU not available (navigator.gpu missing). ' + hint);
    }
    let adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error('No WebGPU adapter. Check chrome://gpu and chrome://flags/#enable-unsafe-webgpu');
    const device = await adapter.requestDevice();
    return { adapter, device };
}

function createBuffer(device, arrOrSize, usage, label) {
    if (arrOrSize instanceof ArrayBuffer || ArrayBuffer.isView(arrOrSize)) {
        const data = arrOrSize instanceof ArrayBuffer
            ? new Uint8Array(arrOrSize)
            : new Uint8Array(arrOrSize.buffer, arrOrSize.byteOffset, arrOrSize.byteLength);
        const buf = device.createBuffer({ size: data.byteLength, usage, mappedAtCreation: true, label });
        new Uint8Array(buf.getMappedRange()).set(data);
        buf.unmap();
        return buf;
    } else {
        return device.createBuffer({ size: arrOrSize, usage, label });
    }
}

function createUniforms(device, imgW, imgH, num, channels) {
    // Params layout must match WGSL struct order and 16-byte alignment rules
    const u32 = new Uint32Array(4);
    u32[0] = imgW;
    u32[1] = imgH;
    u32[2] = num;
    u32[3] = channels;
    return createBuffer(device, u32, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, 'params');
}

function createBindGroupLayoutProject(device) {
    return device.createBindGroupLayout({
        label: 'project-bgl',
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
            { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
            { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        ]
    });
}

function createBindGroupLayoutRaster(device) {
    return device.createBindGroupLayout({
        label: 'raster-bgl',
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
            { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }, // xys (shader declares read_write)
            { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }, // conics (shader declares read_write)
            { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // colors
            { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },          // out_img
        ]
    });
}

async function buildPipelines(device, code) {
    const module = device.createShaderModule({ code, label: 'gsplat-wgsl' });
    const projectBgl = createBindGroupLayoutProject(device);
    const rasterBgl = createBindGroupLayoutRaster(device);

    const projectPipeline = device.createComputePipeline({
        label: 'project-pipeline',
        layout: device.createPipelineLayout({ bindGroupLayouts: [projectBgl] }),
        compute: { module, entryPoint: 'project' }
    });

    const rasterPipeline = device.createComputePipeline({
        label: 'raster-pipeline',
        layout: device.createPipelineLayout({ bindGroupLayouts: [rasterBgl] }),
        compute: { module, entryPoint: 'rasterize' }
    });

    return { module, projectBgl, rasterBgl, projectPipeline, rasterPipeline };
}

function toImageBitmapFromBuffer(device, buffer, width, height, channels) {
    // Convert GPU buffer (H*W*C float32 in [0,1]) to ImageData via a mapped CPU readback
    const byteSize = width * height * channels * 4;
    const readback = device.createBuffer({ size: byteSize, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ, label: 'readback' });

    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(buffer, 0, readback, 0, byteSize);
    device.queue.submit([encoder.finish()]);

    return readback.mapAsync(GPUMapMode.READ).then(() => {
        const f32 = new Float32Array(readback.getMappedRange());
        const rgba = new Uint8ClampedArray(width * height * 4);
        for (let i = 0; i < width * height; i++) {
            const base = i * channels;
            const r = f32[base + 0] ?? 0;
            const g = f32[base + 1] ?? 0;
            const b = f32[base + 2] ?? 0;
            rgba[i * 4 + 0] = Math.max(0, Math.min(255, Math.round(r * 255)));
            rgba[i * 4 + 1] = Math.max(0, Math.min(255, Math.round(g * 255)));
            rgba[i * 4 + 2] = Math.max(0, Math.min(255, Math.round(b * 255)));
            rgba[i * 4 + 3] = 255;
        }
        const imageData = new ImageData(rgba, width, height);
        readback.unmap();
        return createImageBitmap(imageData);
    });
}

async function runRender({ fileBuffer, outCanvas }) {
    const { device } = await ensureWebGPU();
    const wgsl = await loadText('shaders.wgsl');
    const { projectPipeline, rasterPipeline, projectBgl, rasterBgl } = await buildPipelines(device, wgsl);

    // Spawn worker to parse G2D
    const worker = new Worker('g2d-worker.js');
    const parsed = await new Promise((resolve, reject) => {
        worker.onmessage = (ev) => {
            const msg = ev.data;
            if (msg.error) reject(new Error(msg.error));
            else resolve(msg);
        };
        worker.postMessage({ buffer: fileBuffer }, [fileBuffer]);
    });

    const num = parsed.num >>> 0;
    const imgW = parsed.imgW >>> 0;
    const imgH = parsed.imgH >>> 0;
    const ch = parsed.ch >>> 0;

    const paramsBuf = createUniforms(device, imgW, imgH, num, ch);

    // Create GPU buffers
    const meansBuf = createBuffer(device, parsed.xy, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, 'means2d');
    const scalesBuf = createBuffer(device, parsed.scale, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, 'scales2d');
    const rotBuf = createBuffer(device, parsed.rot, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, 'rotation');
    const xysBuf = device.createBuffer({ size: num * 2 * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC, label: 'xys' });
    const conicsBuf = device.createBuffer({ size: num * 3 * 4, usage: GPUBufferUsage.STORAGE, label: 'conics' });
    const colorsBuf = createBuffer(device, parsed.feat, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, 'colors');
    const outImgBuf = device.createBuffer({ size: imgW * imgH * ch * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC, label: 'out_img' });

    // Bind groups
    const projectBg = device.createBindGroup({
        layout: projectBgl,
        entries: [
            { binding: 0, resource: { buffer: paramsBuf } },
            { binding: 1, resource: { buffer: meansBuf } },
            { binding: 2, resource: { buffer: scalesBuf } },
            { binding: 3, resource: { buffer: rotBuf } },
            { binding: 4, resource: { buffer: xysBuf } },
            { binding: 5, resource: { buffer: conicsBuf } },
        ],
        label: 'project-bg'
    });

    const rasterBg = device.createBindGroup({
        layout: rasterBgl,
        entries: [
            { binding: 0, resource: { buffer: paramsBuf } },
            { binding: 4, resource: { buffer: xysBuf } },
            { binding: 5, resource: { buffer: conicsBuf } },
            { binding: 6, resource: { buffer: colorsBuf } },
            { binding: 7, resource: { buffer: outImgBuf } },
        ],
        label: 'raster-bg'
    });

    // Dispatch
    const encoder = device.createCommandEncoder();
    {
        const pass = encoder.beginComputePass();
        pass.setPipeline(projectPipeline);
        pass.setBindGroup(0, projectBg);
        const blocks = Math.ceil(num / 256);
        pass.dispatchWorkgroups(blocks);
        pass.end();
    }
    {
        const pass = encoder.beginComputePass();
        pass.setPipeline(rasterPipeline);
        pass.setBindGroup(0, rasterBg);
        const gx = Math.ceil(imgW / 16);
        const gy = Math.ceil(imgH / 16);
        pass.dispatchWorkgroups(gx, gy, 1);
        pass.end();
    }
    device.queue.submit([encoder.finish()]);

    // Read back and paint on canvas
    const bmp = await toImageBitmapFromBuffer(device, outImgBuf, imgW, imgH, ch);
    outCanvas.width = imgW;
    outCanvas.height = imgH;
    const ctx = outCanvas.getContext('2d');
    ctx.clearRect(0, 0, imgW, imgH);
    ctx.drawImage(bmp, 0, 0);
}

async function main() {
    const fileInput = document.getElementById('g2d');
    const canvas = document.getElementById('out');
    const button = document.getElementById('render');
    const status = document.getElementById('status');

    button.onclick = async () => {
        try {
            status.textContent = 'Parsing and rendering...';
            if (!fileInput.files || fileInput.files.length === 0) throw new Error('Select a .g2d file');
            const file = fileInput.files[0];
            const buf = await file.arrayBuffer();
            await runRender({ fileBuffer: buf, outCanvas: canvas });
            status.textContent = 'Done';
        } catch (e) {
            console.error(e);
            status.textContent = 'Error: ' + e.message;
        }
    };
}

window.addEventListener('DOMContentLoaded', main);


