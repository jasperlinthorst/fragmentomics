(function() {
    window._alignmentUploadResult = null;
    window._tsvUploadResult = null;
    var DEFAULT_SUBSAMPLE = 100000;

    /* ── Utilities ── */
    async function readSlice(f, s, e) {
        var blob = f.slice(s, Math.min(e, f.size));
        return new Uint8Array(await new Response(blob).arrayBuffer());
    }
    function concat(a, b) {
        var r = new Uint8Array(a.length + b.length);
        r.set(a); r.set(b, a.length); return r;
    }
    var _hasDecompressionStream = typeof DecompressionStream !== 'undefined';
    async function gunzip(data) {
        if (!_hasDecompressionStream) throw new Error('DecompressionStream not supported in this browser');
        var ds = new DecompressionStream('gzip');
        var stream = new Blob([data]).stream().pipeThrough(ds);
        var reader = stream.getReader(), chunks = [], n = 0;
        for (;;) { var x = await reader.read(); if (x.done) break; chunks.push(x.value); n += x.value.length; }
        var out = new Uint8Array(n), p = 0;
        for (var i = 0; i < chunks.length; i++) { out.set(chunks[i], p); p += chunks[i].length; }
        return out;
    }
    function setStatus(zone, msg) {
        zone.innerHTML = '<div style="padding:15px;text-align:center;">' + msg + '</div>';
    }

    /* ── BAM: scan BGZF block offsets ── */
    async function scanBAMBlocks(file, onProg) {
        var offsets = [], filePos = 0, BATCH = 4*1024*1024;
        var buf = new Uint8Array(0), bufStart = 0;
        while (filePos < file.size) {
            if (filePos - bufStart + 64 > buf.length) {
                bufStart = filePos;
                buf = await readSlice(file, filePos, filePos + BATCH);
            }
            var lp = filePos - bufStart;
            if (lp + 18 > buf.length) break;
            if (buf[lp] !== 0x1f || buf[lp+1] !== 0x8b) break;
            var xlen = buf[lp+10] | (buf[lp+11] << 8), bsize = -1, pp = lp + 12;
            while (pp < lp + 12 + xlen && pp + 5 < buf.length) {
                if (buf[pp] === 0x42 && buf[pp+1] === 0x43) { bsize = buf[pp+4] | (buf[pp+5] << 8); break; }
                pp += 4 + (buf[pp+2] | (buf[pp+3] << 8));
            }
            if (bsize < 0) break;
            var blockSize = bsize + 1;
            offsets.push({ offset: filePos, size: blockSize });
            filePos += blockSize;
            if (onProg && offsets.length % 500 === 0) onProg(filePos, file.size);
        }
        return offsets;
    }

    /* ── BAM: parse header, return byte offset where records start or -1 ── */
    function parseBAMHeader(buf) {
        if (buf.length < 12) return -1;
        if (buf[0]!==0x42||buf[1]!==0x41||buf[2]!==0x4d||buf[3]!==1) return -1;
        var dv = new DataView(buf.buffer, buf.byteOffset, buf.length);
        var lt = dv.getInt32(4, true), pos = 8 + lt;
        if (pos + 4 > buf.length) return -1;
        var nref = dv.getInt32(pos, true); pos += 4;
        for (var i = 0; i < nref; i++) {
            if (pos + 4 > buf.length) return -1;
            var ln = dv.getInt32(pos, true); pos += 4 + ln;
            if (pos + 4 > buf.length) return -1;
            pos += 4;
        }
        return pos;
    }

    /* ── BAM: count complete records in decompressed data starting at offset ── */
    function countRecordsAt(buf, start) {
        var n = 0, pos = start;
        while (pos + 4 <= buf.length) {
            var dv = new DataView(buf.buffer, buf.byteOffset + pos, 4);
            var rs = dv.getInt32(0, true);
            if (rs <= 0 || pos + 4 + rs > buf.length) break;
            pos += 4 + rs; n++;
        }
        return n;
    }

    var BGZF_EOF = new Uint8Array([
        0x1f,0x8b,0x08,0x04,0x00,0x00,0x00,0x00,0x00,0xff,0x06,0x00,
        0x42,0x43,0x02,0x00,0x1b,0x00,0x03,0x00,0x00,0x00,0x00,0x00,
        0x00,0x00,0x00,0x00]);

    async function subsampleBAM(file, maxReads, zone) {
        setStatus(zone, 'Scanning BAM blocks...');
        var blocks = await scanBAMBlocks(file, function(pos, total) {
            setStatus(zone, 'Scanning BAM blocks... ' + Math.round(pos/total*100) + '%');
        });
        if (blocks.length < 2) return null;
        /* skip EOF block */
        if (blocks[blocks.length-1].size === 28) blocks.pop();

        /* find header extent */
        setStatus(zone, 'Reading BAM header...');
        var headerBlockCount = 0, decompBuf = new Uint8Array(0), headerEnd = -1;
        for (var i = 0; i < blocks.length && i < 100; i++) {
            var bd = await readSlice(file, blocks[i].offset, blocks[i].offset + blocks[i].size);
            decompBuf = concat(decompBuf, await gunzip(bd));
            headerBlockCount = i + 1;
            headerEnd = parseBAMHeader(decompBuf);
            if (headerEnd >= 0) break;
        }
        if (headerEnd < 0) return null;

        /* estimate records per block from ~10 evenly spaced samples */
        var dataStart = headerBlockCount, dataCount = blocks.length - dataStart;
        if (dataCount <= 0) return null;
        var sampleN = Math.min(10, dataCount);
        var step = Math.max(1, Math.floor(dataCount / sampleN));
        var totalSampleRecs = 0, sampledBlocks = 0;
        for (var si = 0; si < sampleN; si++) {
            var idx = dataStart + si * step;
            if (idx >= blocks.length) break;
            var sd = await readSlice(file, blocks[idx].offset, blocks[idx].offset + blocks[idx].size);
            totalSampleRecs += countRecordsAt(await gunzip(sd), 0);
            sampledBlocks++;
        }
        var avgRPB = sampledBlocks > 0 ? totalSampleRecs / sampledBlocks : 300;
        var neededBlocks = Math.ceil(maxReads / avgRPB);

        /* systematic sampling: evenly pick blocks + neighbor for boundary safety */
        var selected = new Set();
        if (neededBlocks >= dataCount) {
            for (var j = dataStart; j < blocks.length; j++) selected.add(j);
        } else {
            var bstep = dataCount / neededBlocks;
            var roff = Math.random() * bstep;
            for (var k = 0; k < neededBlocks; k++) {
                var bi = dataStart + Math.floor(roff + k * bstep);
                if (bi < blocks.length) { selected.add(bi); if (bi+1 < blocks.length) selected.add(bi+1); }
            }
        }
        var sortedSel = Array.from(selected).sort(function(a,b){return a-b;});

        /* read header blocks + selected blocks */
        setStatus(zone, 'Reading ' + sortedSel.length + ' sampled blocks...');
        var parts = [];
        for (var hi = 0; hi < headerBlockCount; hi++) {
            parts.push(await readSlice(file, blocks[hi].offset, blocks[hi].offset + blocks[hi].size));
        }
        for (var ri = 0; ri < sortedSel.length; ri++) {
            var b = blocks[sortedSel[ri]];
            parts.push(await readSlice(file, b.offset, b.offset + b.size));
            if (ri % 50 === 0) setStatus(zone, 'Reading sampled blocks... ' + Math.round(ri/sortedSel.length*100) + '%');
        }
        parts.push(BGZF_EOF);
        var estRecords = Math.round(sortedSel.length * avgRPB);
        return { blob: new Blob(parts), records: estRecords };
    }

    /* ── CRAM: ITF8/LTF8 readers ── */
    function readITF8(buf, pos) {
        if (pos >= buf.length) return null;
        var b = buf[pos];
        if ((b & 0x80) === 0) return {v: b, n: 1};
        if ((b & 0xC0) === 0x80) { if (pos+1>=buf.length) return null; return {v:((b&0x3F)<<8)|buf[pos+1], n:2}; }
        if ((b & 0xE0) === 0xC0) { if (pos+2>=buf.length) return null; return {v:((b&0x1F)<<16)|(buf[pos+1]<<8)|buf[pos+2], n:3}; }
        if ((b & 0xF0) === 0xE0) { if (pos+3>=buf.length) return null; return {v:((b&0x0F)<<24)|(buf[pos+1]<<16)|(buf[pos+2]<<8)|buf[pos+3], n:4}; }
        if (pos+4>=buf.length) return null;
        return {v:((b&0x0F)<<28)|(buf[pos+1]<<20)|(buf[pos+2]<<12)|(buf[pos+3]<<4)|(buf[pos+4]&0x0F), n:5};
    }
    function readLTF8(buf, pos) {
        if (pos >= buf.length) return null;
        var b = buf[pos];
        if ((b & 0x80) === 0) return {v: b, n: 1};
        if ((b & 0xC0) === 0x80) { if(pos+1>=buf.length) return null; return {v:((b&0x3F)<<8)|buf[pos+1], n:2}; }
        if ((b & 0xE0) === 0xC0) { if(pos+2>=buf.length) return null; return {v:((b&0x1F)<<16)|(buf[pos+1]<<8)|buf[pos+2], n:3}; }
        if ((b & 0xF0) === 0xE0) { if(pos+3>=buf.length) return null; return {v:((b&0x0F)<<24)|(buf[pos+1]<<16)|(buf[pos+2]<<8)|buf[pos+3], n:4}; }
        if ((b & 0xF8) === 0xF0) { if(pos+4>=buf.length) return null; return {v:((b&0x07)<<28)|(buf[pos+1]<<20)|(buf[pos+2]<<12)|(buf[pos+3]<<4)|(buf[pos+4]&0x0F), n:5}; }
        /* 6-9 byte: read last 4 bytes for our purposes */
        var nn = (b===0xFF)?9:(b===0xFE)?8:(b>=0xFC)?7:6;
        if(pos+nn-1>=buf.length) return null;
        var val=0; for(var i=1;i<nn;i++) val=(val*256)+buf[pos+i];
        return {v:val, n:nn};
    }

    /* ── CRAM: scan container headers ── */
    async function scanCRAMContainers(file, majorVer) {
        var containers = [], filePos = 26, firstData = true;
        while (filePos < file.size) {
            var hbuf = await readSlice(file, filePos, filePos + 1024);
            if (hbuf.length < 8) break;
            var dv = new DataView(hbuf.buffer, hbuf.byteOffset);
            var cLen = dv.getInt32(0, true), pos = 4;
            if (cLen < 0) break;
            var r = readITF8(hbuf, pos); if(!r) break; pos += r.n; /* refSeqId */
            r = readITF8(hbuf, pos); if(!r) break; pos += r.n; /* startPos */
            r = readITF8(hbuf, pos); if(!r) break; pos += r.n; /* span */
            r = readITF8(hbuf, pos); if(!r) break; var numRecs = r.v; pos += r.n;
            if (majorVer >= 3) { r = readLTF8(hbuf, pos); if(!r) break; pos += r.n; /* recordCounter */ }
            if (majorVer >= 3) { r = readLTF8(hbuf, pos); if(!r) break; pos += r.n; /* bases */ }
            r = readITF8(hbuf, pos); if(!r) break; pos += r.n; /* numBlocks */
            r = readITF8(hbuf, pos); if(!r) break; var nLand = r.v; pos += r.n;
            for (var li = 0; li < nLand; li++) { r = readITF8(hbuf, pos); if(!r) break; pos += r.n; }
            if (!r) break;
            if (majorVer >= 3) pos += 4; /* CRC32 */
            var headerSize = pos, totalSize = headerSize + cLen;
            containers.push({ offset: filePos, totalSize: totalSize, numRecords: numRecs, isHeader: firstData });
            firstData = false;
            filePos += totalSize;
            if (cLen === 0 && numRecs === 0) break; /* EOF container */
        }
        return containers;
    }

    async function subsampleCRAM(file, maxReads, zone) {
        var hdr = await readSlice(file, 0, 26);
        if (hdr.length < 6 || String.fromCharCode(hdr[0],hdr[1],hdr[2],hdr[3]) !== 'CRAM') return null;
        var majorVer = hdr[4];
        setStatus(zone, 'Scanning CRAM containers...');
        var containers = await scanCRAMContainers(file, majorVer);
        if (containers.length < 2) return null;

        /* separate header container (first) from data containers */
        var headerContainer = containers[0];
        var dataCont = containers.slice(1).filter(function(c){ return c.numRecords > 0; });
        if (dataCont.length === 0) return null;

        /* figure out how many containers we need */
        var totalRecs = 0;
        for (var i = 0; i < dataCont.length; i++) totalRecs += dataCont[i].numRecords;
        var needed = [];
        if (totalRecs <= maxReads) {
            needed = dataCont;
        } else {
            /* systematic sampling of containers */
            var pickStep = dataCont.length * (maxReads / totalRecs);
            var nPick = Math.ceil(pickStep);
            var selStep = dataCont.length / nPick;
            var roff = Math.random() * selStep;
            var collected = 0;
            for (var ci = 0; ci < nPick && collected < maxReads; ci++) {
                var idx = Math.floor(roff + ci * selStep);
                if (idx < dataCont.length) { needed.push(dataCont[idx]); collected += dataCont[idx].numRecords; }
            }
        }

        /* read file definition + header container + selected containers */
        setStatus(zone, 'Reading ' + needed.length + ' sampled CRAM containers...');
        var parts = [hdr]; /* file definition */
        parts.push(await readSlice(file, headerContainer.offset, headerContainer.offset + headerContainer.totalSize));
        var estRecs = 0;
        for (var ri = 0; ri < needed.length; ri++) {
            var c = needed[ri];
            parts.push(await readSlice(file, c.offset, c.offset + c.totalSize));
            estRecs += c.numRecords;
            if (ri % 20 === 0) setStatus(zone, 'Reading CRAM containers... ' + Math.round(ri/needed.length*100) + '%');
        }
        return { blob: new Blob(parts), records: estRecs };
    }

    /* ── SAM: fraction-based text sampling ── */
    async function subsampleSAM(file, maxReads, zone) {
        setStatus(zone, 'Estimating SAM size...');
        /* estimate total lines from file size and a sample */
        var sample = await readSlice(file, 0, Math.min(1024*1024, file.size));
        var dec = new TextDecoder();
        var sampleText = dec.decode(sample);
        var sampleLines = sampleText.split('\n');
        var headerLines = 0;
        for (var i = 0; i < sampleLines.length; i++) { if (sampleLines[i].startsWith('@')) headerLines++; else break; }
        var dataLinesInSample = sampleLines.length - headerLines;
        var bytesPerLine = dataLinesInSample > 0 ? (sample.length) / sampleLines.length : 200;
        var estTotal = Math.floor(file.size / bytesPerLine) - headerLines;
        var fraction = estTotal > 0 ? Math.min(1, maxReads / estTotal) : 1;

        setStatus(zone, 'Subsampling SAM (' + Math.round(fraction*100) + '% of ~' + estTotal.toLocaleString() + ' reads)...');
        var CHUNK = 4*1024*1024, offset = 0, leftover = '', kept = 0, outParts = [];
        while (offset < file.size && kept < maxReads * 1.2) {
            var raw = await readSlice(file, offset, offset + CHUNK);
            var text = leftover + dec.decode(raw);
            var lines = text.split('\n');
            leftover = lines.pop();
            var out = '';
            for (var li = 0; li < lines.length; li++) {
                if (lines[li].startsWith('@')) { out += lines[li] + '\n'; continue; }
                if (!lines[li].trim()) continue;
                if (Math.random() < fraction) { out += lines[li] + '\n'; kept++; }
            }
            outParts.push(new TextEncoder().encode(out));
            offset += raw.length;
        }
        return { blob: new Blob(outParts), records: kept };
    }

    /* ── Entry point: detect format and subsample ── */
    async function subsampleAlignment(file, maxReads, zone) {
        if (file.size < 10*1024*1024) return null; /* skip for small files */
        var magic = await readSlice(file, 0, 4);
        try {
            if (magic[0]===0x1f && magic[1]===0x8b) return await subsampleBAM(file, maxReads, zone);
            if (String.fromCharCode(magic[0],magic[1],magic[2],magic[3])==='CRAM') return await subsampleCRAM(file, maxReads, zone);
            var ext = (file.name||'').toLowerCase();
            if (ext.endsWith('.sam')) return await subsampleSAM(file, maxReads, zone);
        } catch(e) { console.warn('Subsampling failed, uploading full file:', e); }
        return null;
    }

    /* ── Upload with optional subsampling ── */
    function doUpload(file, zone, resultVarName, subsample) {
        var origHTML = zone.innerHTML;
        function sendFile(uploadBlob, uploadName, infoMsg) {
            var formData = new FormData();
            formData.append('file', uploadBlob, uploadName);
            setStatus(zone, 'Uploading <b>' + uploadName + '</b>' + (infoMsg||'') +
                '... <span id="_upload-pct">0%</span>');
            var xhr = new XMLHttpRequest();
            xhr.open('POST', '/api/upload-file', true);
            xhr.upload.addEventListener('progress', function(e) {
                if (e.lengthComputable) {
                    var pct = Math.round(e.loaded / e.total * 100);
                    var el = document.getElementById('_upload-pct');
                    if (el) { el.textContent = (pct < 100) ? pct + '%' : 'done, processing...'; }
                }
            });
            xhr.onload = function() {
                zone.innerHTML = origHTML;
                if (xhr.status === 200) {
                    try { window[resultVarName] = JSON.parse(xhr.responseText); }
                    catch(e) { setStatus(zone, '<span style="color:red">Upload parse error</span>'); setTimeout(function(){zone.innerHTML=origHTML;},5000); }
                } else { setStatus(zone, '<span style="color:red">Upload failed ('+xhr.status+')</span>'); setTimeout(function(){zone.innerHTML=origHTML;},5000); }
            };
            xhr.onerror = function() { setStatus(zone, '<span style="color:red">Upload failed (network)</span>'); setTimeout(function(){zone.innerHTML=origHTML;},5000); };
            xhr.send(formData);
        }

        if (!subsample) { sendFile(file, file.name); return; }

        /* read target reads from input if available */
        var targetInput = document.getElementById('subsample-reads-input');
        var maxReads = (targetInput && parseInt(targetInput.value)) || DEFAULT_SUBSAMPLE;

        /* subsample then upload */
        setStatus(zone, 'Preparing <b>' + file.name + '</b>...');
        subsampleAlignment(file, maxReads, zone).then(function(result) {
            if (result) {
                var sizeMB = (result.blob.size / 1024 / 1024).toFixed(1);
                sendFile(result.blob, file.name,
                    ' (~' + result.records.toLocaleString() + ' reads, ' + sizeMB + ' MB)');
            } else {
                sendFile(file, file.name);
            }
        }).catch(function(e) {
            console.warn('Subsample error:', e);
            sendFile(file, file.name);
        });
    }

    /* ── Drop zone initialization ── */
    function initDropZone(zoneId, resultVarName, subsample) {
        var zone = document.getElementById(zoneId);
        if (!zone || zone._dropInit) return;
        zone._dropInit = true;
        zone.style.cursor = 'pointer';
        var fileInput = document.createElement('input');
        fileInput.type = 'file'; fileInput.style.display = 'none';
        zone.appendChild(fileInput);
        zone.addEventListener('click', function(e) { e.preventDefault(); e.stopPropagation(); fileInput.click(); });
        zone.addEventListener('dragover', function(e) { e.preventDefault(); e.stopPropagation(); zone.style.borderColor='#2196F3'; zone.style.backgroundColor='#e3f2fd'; });
        zone.addEventListener('dragleave', function(e) { e.preventDefault(); e.stopPropagation(); zone.style.borderColor=''; zone.style.backgroundColor=''; });
        zone.addEventListener('drop', function(e) {
            e.preventDefault(); e.stopPropagation(); zone.style.borderColor=''; zone.style.backgroundColor='';
            if (e.dataTransfer.files.length > 0) doUpload(e.dataTransfer.files[0], zone, resultVarName, subsample);
        });
        fileInput.addEventListener('change', function() {
            if (fileInput.files.length > 0) { doUpload(fileInput.files[0], zone, resultVarName, subsample); fileInput.value = ''; }
        });
    }

    setInterval(function() {
        initDropZone('alignment-dropzone', '_alignmentUploadResult', true);
        initDropZone('tsv-dropzone', '_tsvUploadResult', false);
    }, 500);
})();
