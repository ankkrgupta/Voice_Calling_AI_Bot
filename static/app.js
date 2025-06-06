// // ───────────────────────────────────────────────────────────────────────────────
// //                                   app.js                                  
// // ───────────────────────────────────────────────────────────────────────────────
// //
// // This main script replaces createScriptProcessor with an AudioWorkletNode.
// // Incoming TTS Int16@48k chunks are converted to Float32@48k and then—if
// // audioContext.sampleRate ≠ 48000—resampled to the context’s rate before
// // posting to the worklet. The rest of the flow is unchanged.
// //
// // Copy this file into `static/app.js` (replacing your old version). Everything
// // except the small resampling block in handleBinaryFrame() is exactly as before.
// // ───────────────────────────────────────────────────────────────────────────────

// let audioContext = null;
// let playbackRate = null;           // AudioContext.sampleRate (48k, 16k, etc.)
// let playProcessorNode = null;       // AudioWorkletNode("playback-processor")

// let scriptNodeSender = null;        // ScriptProcessorNode for mic→WS
// let micStream = null;
// let ws = null;                      // WebSocket instance

// let float32Queue = [];              // FIFO of Float32 mic samples
// const TARGET_SAMPLE_RATE = 16000;
// let micSampleRate = 48000;

// // When all queued samples finish, playbackEndTime marks the AudioContext time + 50 ms safety
// let playbackEndTime = 0;

// // ─── On load: create AudioContext + load Worklet ───────────────────────────────
// window.addEventListener("load", async () => {
//   audioContext = new (window.AudioContext || window.webkitAudioContext)();
//   await audioContext.resume();

//   playbackRate = audioContext.sampleRate;  // e.g. 48000, 16000, etc.
//   console.log("[DEBUG] audioContext.sampleRate =", playbackRate);

//   // 1) Add the playback-processor module
//   try {
//     await audioContext.audioWorklet.addModule("/static/playback-processor.js");
//   } catch (err) {
//     console.error("[ERROR] Could not load playback-processor.js:", err);
//     return;
//   }

//   // 2) Create the AudioWorkletNode
//   playProcessorNode = new AudioWorkletNode(audioContext, "playback-processor");
//   // Connect it to the destination so it actually outputs sound:
//   playProcessorNode.connect(audioContext.destination);

//   console.log("[DEBUG] AudioWorkletNode(playback-processor) created");
// });

// // ─── Downsample or Upsample Float32Array [srcRate → 16 kHz] ─────────────────────────
// function downsampleBuffer(buffer, srcRate) {
//   const ratio = srcRate / TARGET_SAMPLE_RATE;
//   const newLength = Math.floor(buffer.length / ratio);
//   const result = new Float32Array(newLength);
//   for (let i = 0; i < newLength; i++) {
//     const start = Math.floor(i * ratio);
//     const end = Math.min(buffer.length, Math.floor((i + 1) * ratio));
//     let sum = 0, count = 0;
//     for (let j = start; j < end; j++) {
//       sum += buffer[j];
//       count++;
//     }
//     result[i] = count > 0 ? sum / count : 0;
//   }
//   return result;
// }

// // ─── Resample Float32Array [48000 → playbackRate] (linear interpolation) ──────────
// function resample48kToPlayback(buffer48k) {
//   if (playbackRate === 48000) {
//     return buffer48k.slice(); // shallow copy if no resampling needed
//   }
//   const targetLen = Math.round((buffer48k.length * playbackRate) / 48000);
//   const result = new Float32Array(targetLen);
//   const ratio = 48000 / playbackRate;
//   for (let i = 0; i < targetLen; i++) {
//     const idx = i * ratio;
//     const i0 = Math.floor(idx);
//     const i1 = Math.min(buffer48k.length - 1, i0 + 1);
//     const w = idx - i0;
//     result[i] = (1 - w) * buffer48k[i0] + w * buffer48k[i1];
//   }
//   return result;
// }

// // ─── Convert Float32Array [–1..1] → Int16Array ─────────────────────────────────
// function floatToInt16(floatBuffer) {
//   const int16 = new Int16Array(floatBuffer.length);
//   for (let i = 0; i < floatBuffer.length; i++) {
//     const s = Math.max(-1, Math.min(1, floatBuffer[i]));
//     int16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
//   }
//   return int16;
// }

// // ─── When we receive an Int16@48k TTS chunk, convert, resample, and send ──────────
// function handleBinaryFrame(arrayBuffer) {
//   // 1) Read raw 16-bit PCM @ 48 kHz
//   const pcm16 = new Int16Array(arrayBuffer);
//   if (!pcm16.length) return;

//   // 2) Convert to Float32 @ 48 kHz
//   const floats48k = new Float32Array(pcm16.length);
//   for (let i = 0; i < pcm16.length; i++) {
//     floats48k[i] = pcm16[i] / 32768;
//   }

//   // ─── NEW: Resample from 48 kHz → playbackRate, if needed ───────────────────
//   let toSend;
//   if (playbackRate !== 48000) {
//     // Linear‐interpolate from 48 kHz → playbackRate
//     toSend = resample48kToPlayback(floats48k);
//   } else {
//     toSend = floats48k;
//   }

//   // 3) Post that (possibly resampled) Float32 chunk to the worklet’s port
//   playProcessorNode.port.postMessage(toSend);

//   // 4) Recompute playbackEndTime:
//   //    We assume “toSend.length” samples will be drained in real time at playbackRate.
//   const now = audioContext.currentTime;
//   playbackEndTime = now + (toSend.length / playbackRate);

//   console.log(
//     "[DEBUG] TTS chunk arrived (orig", pcm16.length, "samples → resampled", 
//     toSend.length, "samples). Approx unmute at", playbackEndTime.toFixed(3)
//   );
// }

// // ─── Build WebSocket and attach handlers ───────────────────────────────────────
// function setupWebSocket() {
//   const protocol = window.location.protocol === "https:" ? "wss" : "ws";
//   ws = new WebSocket(`${protocol}://${window.location.host}/ws`);
//   ws.binaryType = "arraybuffer";

//   ws.onopen = () => console.log("[WS] Connection opened");
//   ws.onerror = (err) => console.error("[WS] Error:", err);
//   ws.onclose = () => console.log("[WS] Connection closed");

//   ws.onmessage = (evt) => {
//     if (evt.data instanceof ArrayBuffer) {
//       // Incoming TTS chunk → handle (convert/resample/send & update end time)
//       handleBinaryFrame(evt.data);
//     } else if (evt.data instanceof Blob) {
//       // Some browsers wrap bytes in a Blob
//       evt.data
//         .arrayBuffer()
//         .then((ab) => handleBinaryFrame(ab))
//         .catch((e) => console.error("[WS] Blob→ArrayBuffer error:", e));
//     } else {
//       // JSON control message: transcript or token
//       try {
//         const msg = JSON.parse(evt.data);
//         if (msg.type === "transcript") {
//           const label = msg.final ? "FINAL" : "INTERIM";
//           document.getElementById("transcripts").textContent +=
//             `\nTRANSCRIPT [${label}]: ${msg.text}\n`;
//         } else if (msg.type === "token") {
//           document.getElementById("transcripts").textContent += msg.token;
//         }
//       } catch {
//         console.warn("[WS] Non-JSON message:", evt.data);
//       }
//     }
//   };

//   return ws;
// }

// // ─── Start capturing mic → WS, skip when currentTime < playbackEndTime ─────────
// function startMicStreaming(ws) {
//   const bufferSize = 4096;
//   scriptNodeSender = audioContext.createScriptProcessor(bufferSize, 1, 1);
//   const micSource = audioContext.createMediaStreamSource(micStream);
//   micSource.connect(scriptNodeSender);

//   scriptNodeSender.onaudioprocess = (event) => {
//     // If the currentTime is still before playbackEndTime + 50 ms, drop mic
//     if (audioContext.currentTime < playbackEndTime + 0.05) {
//       return;
//     }

//     // Otherwise, capture mic floats as before
//     const inData = event.inputBuffer.getChannelData(0);
//     float32Queue.push(new Float32Array(inData));

//     const total = float32Queue.reduce((sum, arr) => sum + arr.length, 0);
//     const needed = Math.ceil((micSampleRate / TARGET_SAMPLE_RATE) * 320);
//     if (total < needed) return;

//     // Merge queued floats
//     const merged = new Float32Array(total);
//     let off = 0;
//     float32Queue.forEach((chunk) => {
//       merged.set(chunk, off);
//       off += chunk.length;
//     });

//     // Downsample merged@micSampleRate → 16 kHz
//     const down16k = downsampleBuffer(merged, micSampleRate);

//     // Send 320‐sample Int16 frames
//     let i = 0;
//     while (i + 320 <= down16k.length) {
//       const slice = down16k.subarray(i, i + 320);
//       const int16 = floatToInt16(slice);
//       if (ws.readyState === WebSocket.OPEN) {
//         ws.send(int16.buffer);
//       }
//       i += 320;
//     }

//     // Keep leftover for next round
//     const leftoverIn = Math.round((down16k.length - i) * (micSampleRate / TARGET_SAMPLE_RATE));
//     const leftover = merged.subarray(merged.length - leftoverIn);
//     float32Queue = leftoverIn > 0 ? [leftover] : [];
//   };

//   // Connect to destination so the node stays alive (no output)
//   scriptNodeSender.connect(audioContext.destination);
//   console.log("[MIC] Mic streaming started");
// }

// // ─── Start everything when user clicks “Start” ─────────────────────────────────
// async function startStreaming() {
//   // 1) Resume AudioContext if needed
//   if (audioContext.state === "suspended") {
//     await audioContext.resume();
//     console.log("[DEBUG] audioContext resumed; state =", audioContext.state);
//   }

//   // 2) Request microphone
//   try {
//     micStream = await navigator.mediaDevices.getUserMedia({
//       audio: { echoCancellation: true, noiseSuppression: true },
//     });
//     micSampleRate = audioContext.sampleRate;
//     console.log("[DEBUG] Mic sampleRate =", micSampleRate);
//   } catch (err) {
//     console.error("[UI] getUserMedia error:", err);
//     return;
//   }

//   // 3) Open WebSocket  
//   const socket = setupWebSocket();

//   // 4) Once WS is open, start mic streaming
//   socket.addEventListener("open", () => {
//     console.log("[WS] Ready → starting mic streaming");
//     startMicStreaming(socket);
//   });
// }

// // ─── Wire Start/Stop button ───────────────────────────────────────────────────
// document.addEventListener("DOMContentLoaded", () => {
//   const btn = document.getElementById("startStopBtn");
//   let streaming = false;

//   btn.addEventListener("click", async () => {
//     if (!streaming) {
//       btn.textContent = "Stop";
//       streaming = true;
//       await startStreaming();
//     } else {
//       btn.textContent = "Start";
//       streaming = false;

//       // Tear down mic processing
//       if (scriptNodeSender) {
//         scriptNodeSender.disconnect();
//         scriptNodeSender.onaudioprocess = null;
//         scriptNodeSender = null;
//       }
//       if (micStream) {
//         micStream.getTracks().forEach((t) => t.stop());
//         micStream = null;
//       }
//       float32Queue = [];

//       // Close WebSocket
//       if (ws && ws.readyState === WebSocket.OPEN) {
//         ws.close();
//       }

//       // Reset playback queue + end time
//       playQueue = [];
//       playbackEndTime = 0;
//     }
//   });
// });


/* Added Multiple Language*/


// // ───────────────────────────────────────────────────────────────────────────────
// //                                   app.js                                  
// // ───────────────────────────────────────────────────────────────────────────────
// //
// // Uses an AudioWorkletNode for TTS playback and a ScriptProcessorNode for mic capture.
// // You can pick “en-IN” or “multi” *before* clicking “Start the Conversation.” That
// // choice is sent as ?lang=<…> to the WebSocket URL. Once “Start” is clicked, an
// // informational message appears showing which language is in effect and how to change it.
// //
// // Copy this file into `static/app.js`, making sure your HTML loads it last.
// // ───────────────────────────────────────────────────────────────────────────────

// let audioContext = null;
// let playbackRate = null;           // AudioContext.sampleRate (48k, 16k, etc.)
// let playProcessorNode = null;       // AudioWorkletNode("playback-processor")

// let scriptNodeSender = null;        // ScriptProcessorNode for mic→WS
// let micStream = null;
// let ws = null;                      // WebSocket instance

// let float32Queue = [];              // FIFO of Float32 mic samples
// const TARGET_SAMPLE_RATE = 16000;
// let micSampleRate = 48000;

// // When all queued samples finish, playbackEndTime marks the AudioContext time + 50 ms safety
// let playbackEndTime = 0;

// // ─── LANGUAGE STATE ────────────────────────────────────────────────────────────
// // Start with English‐IN by default. Clicking a .langBtn updates this variable,
// // and the “selected” CSS class gets toggled for visual feedback.
// let selectedLanguage = "en-IN";

// // ─── On load: create AudioContext + load Worklet ───────────────────────────────
// window.addEventListener("load", async () => {
//   audioContext = new (window.AudioContext || window.webkitAudioContext)();
//   await audioContext.resume();

//   playbackRate = audioContext.sampleRate;  // e.g. 48000, 16000, etc.
//   console.log("[DEBUG] audioContext.sampleRate =", playbackRate);

//   // 1) Add the playback-processor module
//   try {
//     await audioContext.audioWorklet.addModule("/static/playback-processor.js");
//   } catch (err) {
//     console.error("[ERROR] Could not load playback-processor.js:", err);
//     return;
//   }

//   // 2) Create the AudioWorkletNode
//   playProcessorNode = new AudioWorkletNode(audioContext, "playback-processor");
//   // Connect it to the destination so it actually outputs sound:
//   playProcessorNode.connect(audioContext.destination);

//   console.log("[DEBUG] AudioWorkletNode(playback-processor) created");

//   // ─── Wire up language buttons so that clicking one toggles “selected” class ───
//   document.querySelectorAll(".langBtn").forEach(btn => {
//     btn.addEventListener("click", () => {
//       // Remove “selected” from all, then add to the one clicked
//       document.querySelectorAll(".langBtn").forEach(b => b.classList.remove("selected"));
//       btn.classList.add("selected");

//       // Update our local state
//       selectedLanguage = btn.getAttribute("data-lang");
//       console.log(`[UI] Selected language = ${selectedLanguage}`);
//     });
//   });

//   // ─── Wire up Start/Stop button ──────────────────────────────────────────────
//   const startStopBtn = document.getElementById("startStopBtn");
//   let streaming = false;

//   startStopBtn.addEventListener("click", async () => {
//     if (!streaming) {
//       // 1) Change to “Stop” style
//       startStopBtn.textContent = "Stop the Conversation";
//       startStopBtn.classList.add("stop");
//       streaming = true;

//       // 2) Show informational message about current language
//       const infoDiv = document.getElementById("info");
//       if (selectedLanguage === "en-IN") {
//         infoDiv.textContent = 
//           "Current language of conversation is English. " +
//           "If you want to shift to Multi, select it above, then stop the ongoing conversation and click on start conversation button again.";
//       } else {
//         infoDiv.textContent = 
//           "Current language of conversation is Multi (Hinglish). " +
//           "If you want to shift to English, select it above, then stop the ongoing conversation and click on start conversation button again.";
//       }

//       // 3) Kick off WebSocket + mic capture
//       await startStreaming();
//     } else {
//       // Stop the conversation
//       startStopBtn.textContent = "Start the Conversation";
//       startStopBtn.classList.remove("stop");
//       streaming = false;

//       // Clear the informational message
//       document.getElementById("info").textContent = "";

//       // Tear down mic processing
//       if (scriptNodeSender) {
//         scriptNodeSender.disconnect();
//         scriptNodeSender.onaudioprocess = null;
//         scriptNodeSender = null;
//       }
//       if (micStream) {
//         micStream.getTracks().forEach((t) => t.stop());
//         micStream = null;
//       }
//       float32Queue = [];

//       // Close WebSocket (if open)
//       if (ws && ws.readyState === WebSocket.OPEN) {
//         ws.close();
//       }

//       // Reset playback queue + end time
//       playbackEndTime = 0;
//     }
//   });
// });

// // ─── Downsample or Upsample Float32Array [srcRate → 16 kHz] ─────────────────────────
// function downsampleBuffer(buffer, srcRate) {
//   const ratio = srcRate / TARGET_SAMPLE_RATE;
//   const newLength = Math.floor(buffer.length / ratio);
//   const result = new Float32Array(newLength);
//   for (let i = 0; i < newLength; i++) {
//     const start = Math.floor(i * ratio);
//     const end = Math.min(buffer.length, Math.floor((i + 1) * ratio));
//     let sum = 0, count = 0;
//     for (let j = start; j < end; j++) {
//       sum += buffer[j];
//       count++;
//     }
//     result[i] = count > 0 ? sum / count : 0;
//   }
//   return result;
// }

// // ─── Resample Float32Array [48000 → playbackRate] (linear interpolation) ──────────
// function resample48kToPlayback(buffer48k) {
//   if (playbackRate === 48000) {
//     return buffer48k.slice(); // shallow copy if no resampling needed
//   }
//   const targetLen = Math.round((buffer48k.length * playbackRate) / 48000);
//   const result = new Float32Array(targetLen);
//   const ratio = 48000 / playbackRate;
//   for (let i = 0; i < targetLen; i++) {
//     const idx = i * ratio;
//     const i0 = Math.floor(idx);
//     const i1 = Math.min(buffer48k.length - 1, i0 + 1);
//     const w = idx - i0;
//     result[i] = (1 - w) * buffer48k[i0] + w * buffer48k[i1];
//   }
//   return result;
// }

// // ─── Convert Float32Array [–1..1] → Int16Array ─────────────────────────────────
// function floatToInt16(floatBuffer) {
//   const int16 = new Int16Array(floatBuffer.length);
//   for (let i = 0; i < floatBuffer.length; i++) {
//     const s = Math.max(-1, Math.min(1, floatBuffer[i]));
//     int16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
//   }
//   return int16;
// }

// // ─── When we receive an Int16@48k TTS chunk, convert, resample, and send ──────────
// function handleBinaryFrame(arrayBuffer) {
//   // 1) Read raw 16-bit PCM @ 48 kHz
//   const pcm16 = new Int16Array(arrayBuffer);
//   if (!pcm16.length) return;

//   // 2) Convert to Float32 @ 48 kHz
//   const floats48k = new Float32Array(pcm16.length);
//   for (let i = 0; i < pcm16.length; i++) {
//     floats48k[i] = pcm16[i] / 32768;
//   }

//   // ─── Resample from 48 kHz → playbackRate, if needed ───────────────────
//   let toSend;
//   if (playbackRate !== 48000) {
//     toSend = resample48kToPlayback(floats48k);
//   } else {
//     toSend = floats48k;
//   }

//   // 3) Post that (possibly resampled) Float32 chunk to the worklet’s port
//   playProcessorNode.port.postMessage(toSend);

//   // 4) Recompute playbackEndTime:
//   //    We assume “toSend.length” samples will be drained in real time at playbackRate.
//   const now = audioContext.currentTime;
//   playbackEndTime = now + (toSend.length / playbackRate);

//   console.log(
//     "[DEBUG] TTS chunk arrived (orig", pcm16.length, "samples → resampled",
//     toSend.length, "samples). Approx unmute at", playbackEndTime.toFixed(3)
//   );
// }

// // ─── Build WebSocket and attach handlers ───────────────────────────────────────
// function setupWebSocket() {
//   // Include chosenLanguage as a query parameter
//   const protocol = window.location.protocol === "https:" ? "wss" : "ws";
//   const url = `${protocol}://${window.location.host}/ws?lang=${encodeURIComponent(selectedLanguage)}`;
//   ws = new WebSocket(url);
//   ws.binaryType = "arraybuffer";

//   ws.onopen = () => {
//     console.log("[WS] Connection opened");
//     console.log(`[UI] WS opened with ?lang=${selectedLanguage}`);
//   };

//   ws.onerror = (err) => console.error("[WS] Error:", err);
//   ws.onclose = () => console.log("[WS] Connection closed");

//   ws.onmessage = (evt) => {
//     if (evt.data instanceof ArrayBuffer) {
//       // Incoming TTS chunk → handle (convert/resample/send & update end time)
//       handleBinaryFrame(evt.data);
//     } else if (evt.data instanceof Blob) {
//       // Some browsers wrap bytes in a Blob
//       evt.data
//         .arrayBuffer()
//         .then((ab) => handleBinaryFrame(ab))
//         .catch((e) => console.error("[WS] Blob→ArrayBuffer error:", e));
//     } else {
//       // JSON control message: transcript or token
//       try {
//         const msg = JSON.parse(evt.data);
//         if (msg.type === "transcript") {
//           const label = msg.final ? "FINAL" : "INTERIM";
//           document.getElementById("transcripts").textContent +=
//             `\nTRANSCRIPT [${label}]: ${msg.text}\n`;
//         } else if (msg.type === "token") {
//           document.getElementById("transcripts").textContent += msg.token;
//         }
//       } catch {
//         console.warn("[WS] Non-JSON message:", evt.data);
//       }
//     }
//   };

//   return ws;
// }

// // ─── Start capturing mic → WS, skip when currentTime < playbackEndTime ─────────
// function startMicStreaming(ws) {
//   const bufferSize = 4096;
//   scriptNodeSender = audioContext.createScriptProcessor(bufferSize, 1, 1);
//   const micSource = audioContext.createMediaStreamSource(micStream);
//   micSource.connect(scriptNodeSender);

//   scriptNodeSender.onaudioprocess = (event) => {
//     // If the currentTime is still before playbackEndTime + 50 ms, drop mic
//     if (audioContext.currentTime < playbackEndTime + 0.05) {
//       return;
//     }

//     // Otherwise, capture mic floats as before
//     const inData = event.inputBuffer.getChannelData(0);
//     float32Queue.push(new Float32Array(inData));

//     const total = float32Queue.reduce((sum, arr) => sum + arr.length, 0);
//     const needed = Math.ceil((micSampleRate / TARGET_SAMPLE_RATE) * 320);
//     if (total < needed) return;

//     // Merge queued floats
//     const merged = new Float32Array(total);
//     let off = 0;
//     float32Queue.forEach((chunk) => {
//       merged.set(chunk, off);
//       off += chunk.length;
//     });

//     // Downsample merged@micSampleRate → 16 kHz
//     const down16k = downsampleBuffer(merged, micSampleRate);

//     // Send 320‐sample Int16 frames
//     let i = 0;
//     while (i + 320 <= down16k.length) {
//       const slice = down16k.subarray(i, i + 320);
//       const int16 = floatToInt16(slice);
//       if (ws.readyState === WebSocket.OPEN) {
//         ws.send(int16.buffer);
//       }
//       i += 320;
//     }

//     // Keep leftover for next round
//     const leftoverIn = Math.round((down16k.length - i) * (micSampleRate / TARGET_SAMPLE_RATE));
//     const leftover = merged.subarray(merged.length - leftoverIn);
//     float32Queue = leftoverIn > 0 ? [leftover] : [];
//   };

//   // Connect to destination so the node stays alive (no output)
//   scriptNodeSender.connect(audioContext.destination);
//   console.log("[MIC] Mic streaming started");
// }

// // ─── Start everything when user clicks “Start” ─────────────────────────────────
// async function startStreaming() {
//   // 1) Resume AudioContext if needed
//   if (audioContext.state === "suspended") {
//     await audioContext.resume();
//     console.log("[DEBUG] audioContext resumed; state =", audioContext.state);
//   }

//   // 2) Request microphone
//   try {
//     micStream = await navigator.mediaDevices.getUserMedia({
//       audio: { echoCancellation: true, noiseSuppression: true },
//     });
//     micSampleRate = audioContext.sampleRate;
//     console.log("[DEBUG] Mic sampleRate =", micSampleRate);
//   } catch (err) {
//     console.error("[UI] getUserMedia error:", err);
//     return;
//   }

//   // 3) Open WebSocket (includes ?lang=<selectedLanguage>)
//   const socket = setupWebSocket();

//   // 4) Once WS is open, start mic streaming
//   socket.addEventListener("open", () => {
//     console.log("[WS] Ready → starting mic streaming");
//     startMicStreaming(socket);
//   });
// }


// ───────────────────────────────────────────────────────────────────────────────
//                                   app.js                                  
// ───────────────────────────────────────────────────────────────────────────────
//
// We used to create an AudioContext on page load, but that is blocked unless
// it is inside a user gesture. Now we defer creating/resuming the AudioContext
// (and loading the worklet) until the user actually clicks "Start the Conversation."
// Everything else (language toggling, mic streaming logic, TTS playback) remains
// unchanged.
//
// Copy this file into `static/app.js`, replacing your old version. 
// ───────────────────────────────────────────────────────────────────────────────

let audioContext = null;
let playbackRate = null;           // Will be audioContext.sampleRate once created
let playProcessorNode = null;      // AudioWorkletNode("playback-processor")

let scriptNodeSender = null;       // ScriptProcessorNode for mic→WS
let micStream = null;
let ws = null;                     // WebSocket instance

let float32Queue = [];             // FIFO of Float32 mic samples
const TARGET_SAMPLE_RATE = 16000;
let micSampleRate = 48000;

// When all queued TTS samples finish, playbackEndTime ≈ audioContext.currentTime + (queuedSamples / sampleRate)
let playbackEndTime = 0;

// ─── LANGUAGE STATE ────────────────────────────────────────────────────────────
// Default to English‐IN. Clicking a language button simply updates this.
let selectedLanguage = "en-IN";

// ─── Flag to know if AudioContext + Worklet is already set up ──────────────────
let audioReady = false;

// ─── Wire up language toggle buttons immediately ───────────────────────────────
window.addEventListener("load", () => {
  document.querySelectorAll(".langBtn").forEach(btn => {
    btn.addEventListener("click", () => {
      // Remove 'selected' from all, then add to clicked
      document.querySelectorAll(".langBtn").forEach(x => x.classList.remove("selected"));
      btn.classList.add("selected");

      const chosen = btn.getAttribute("data-lang");
      selectedLanguage = chosen;
      console.log(`[UI] Language set to ${chosen} (will apply on next Start)`);
    });
  });
});

// ─── Downsample Float32Array [srcRate → 16 kHz] ──────────────────────────────────
function downsampleBuffer(buffer, srcRate) {
  const ratio = srcRate / TARGET_SAMPLE_RATE;
  const newLength = Math.floor(buffer.length / ratio);
  const result = new Float32Array(newLength);
  for (let i = 0; i < newLength; i++) {
    const start = Math.floor(i * ratio);
    const end = Math.min(buffer.length, Math.floor((i + 1) * ratio));
    let sum = 0, count = 0;
    for (let j = start; j < end; j++) {
      sum += buffer[j];
      count++;
    }
    result[i] = count > 0 ? sum / count : 0;
  }
  return result;
}

// ─── Resample Float32Array [48000 → playbackRate] (linear interpolation) ──────────
function resample48kToPlayback(buffer48k) {
  if (playbackRate === 48000) {
    return buffer48k.slice();
  }
  const targetLen = Math.round((buffer48k.length * playbackRate) / 48000);
  const result = new Float32Array(targetLen);
  const ratio = 48000 / playbackRate;
  for (let i = 0; i < targetLen; i++) {
    const idx = i * ratio;
    const i0 = Math.floor(idx);
    const i1 = Math.min(buffer48k.length - 1, i0 + 1);
    const w = idx - i0;
    result[i] = (1 - w) * buffer48k[i0] + w * buffer48k[i1];
  }
  return result;
}

// ─── Convert Float32Array [–1..1] → Int16Array ─────────────────────────────────
function floatToInt16(floatBuffer) {
  const int16 = new Int16Array(floatBuffer.length);
  for (let i = 0; i < floatBuffer.length; i++) {
    const s = Math.max(-1, Math.min(1, floatBuffer[i]));
    int16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return int16;
}

// ─── When we receive an Int16@48k TTS chunk, convert/resample/send ──────────────
function handleBinaryFrame(arrayBuffer) {
  // 1) Read raw 16-bit PCM @ 48 kHz
  const pcm16 = new Int16Array(arrayBuffer);
  if (!pcm16.length) return;

  // 2) Convert to Float32 @ 48 kHz
  const floats48k = new Float32Array(pcm16.length);
  for (let i = 0; i < pcm16.length; i++) {
    floats48k[i] = pcm16[i] / 32768;
  }

  // 3) Resample from 48 kHz → playbackRate, if needed
  let toSend;
  if (playbackRate !== 48000) {
    toSend = resample48kToPlayback(floats48k);
  } else {
    toSend = floats48k;
  }

  // 4) Post that (possibly resampled) Float32 chunk to the worklet’s port
  playProcessorNode.port.postMessage(toSend);

  // 5) Recompute playbackEndTime (so mic can mute/unmute correctly)
  const now = audioContext.currentTime;
  playbackEndTime = now + (toSend.length / playbackRate);

  console.log(
    "[DEBUG] TTS chunk arrived (orig", pcm16.length, "samples → resampled",
    toSend.length, "samples). Approx unmute at", playbackEndTime.toFixed(3)
  );
}

// ─── Build WebSocket and attach handlers ───────────────────────────────────────
function setupWebSocket() {
  // Include selectedLanguage as a query parameter
  const protocol = window.location.protocol === "https:" ? "wss" : "ws";
  const url = `${protocol}://${window.location.host}/ws?lang=${encodeURIComponent(selectedLanguage)}`;
  ws = new WebSocket(url);
  ws.binaryType = "arraybuffer";

  ws.onopen = () => {
    console.log("[WS] Connection opened");
    console.log(`[UI] WS opened with ?lang=${selectedLanguage}`);

    // Once the socket is open, show the status message
    const statusEl = document.getElementById("statusMessage");
    if (selectedLanguage === "en-IN") {
      statusEl.textContent =
        "Current language of conversation is English (IN). " +
        "If you want to shift to Multi, select it above, then Stop and Start again.";
    } else if (selectedLanguage === "multi") {
      statusEl.textContent =
        "Current language of conversation is Multi. " +
        "If you want to shift to English (IN), select it above, then Stop and Start again.";
    } else {
      statusEl.textContent = `Current language: ${selectedLanguage}.`;
    }
  };

  ws.onerror = (err) => console.error("[WS] Error:", err);
  ws.onclose = () => console.log("[WS] Connection closed");

  // ─── Handle incoming WS messages ─────────────────────────────────────────────
  ws.onmessage = (evt) => {
    if (evt.data instanceof ArrayBuffer) {
      // TTS chunk: convert/resample/send & update playbackEndTime
      handleBinaryFrame(evt.data);
    } else if (evt.data instanceof Blob) {
      // Some browsers wrap bytes in a Blob
      evt.data
        .arrayBuffer()
        .then((ab) => handleBinaryFrame(ab))
        .catch((e) => console.error("[WS] Blob→ArrayBuffer error:", e));
    } else {
      // JSON control message: transcript or token
      try {
        const msg = JSON.parse(evt.data);
        if (msg.type === "transcript") {
          const label = msg.final ? "FINAL" : "INTERIM";
          document.getElementById("transcripts").textContent +=
            `\nTRANSCRIPT [${label}]: ${msg.text}\n`;
        } else if (msg.type === "token") {
          document.getElementById("transcripts").textContent += msg.token;
        }
      } catch {
        console.warn("[WS] Non-JSON message:", evt.data);
      }
    }
  };

  return ws;
}

// ─── Start mic capture → WS, skipping while TTS is playing ────────────────────
function startMicStreaming(ws) {
  const bufferSize = 4096;
  scriptNodeSender = audioContext.createScriptProcessor(bufferSize, 1, 1);
  const micSource = audioContext.createMediaStreamSource(micStream);
  micSource.connect(scriptNodeSender);

  scriptNodeSender.onaudioprocess = (event) => {
    // If currentTime < playbackEndTime + 50 ms, drop mic
    if (audioContext.currentTime < playbackEndTime + 0.05) {
      return;
    }

    // Otherwise, capture mic floats
    const inData = event.inputBuffer.getChannelData(0);
    float32Queue.push(new Float32Array(inData));

    const total = float32Queue.reduce((sum, arr) => sum + arr.length, 0);
    const needed = Math.ceil((micSampleRate / TARGET_SAMPLE_RATE) * 320);
    if (total < needed) return;

    // Merge queued floats
    const merged = new Float32Array(total);
    let off = 0;
    float32Queue.forEach((chunk) => {
      merged.set(chunk, off);
      off += chunk.length;
    });

    // Downsample merged@micSampleRate → 16 kHz
    const down16k = downsampleBuffer(merged, micSampleRate);

    // Send out 320-sample (20 ms) Int16 frames
    let i = 0;
    while (i + 320 <= down16k.length) {
      const slice = down16k.subarray(i, i + 320);
      const int16 = floatToInt16(slice);
      if (ws.readyState === WebSocket.OPEN) {
        ws.send(int16.buffer);
      }
      i += 320;
    }

    // Keep leftover
    const leftoverIn = Math.round((down16k.length - i) * (micSampleRate / TARGET_SAMPLE_RATE));
    const leftover = merged.subarray(merged.length - leftoverIn);
    float32Queue = leftoverIn > 0 ? [leftover] : [];
  };

  // Connect to destination so the node stays alive (no output)
  scriptNodeSender.connect(audioContext.destination);
  console.log("[MIC] Mic streaming started");
}

// ─── Initialize AudioContext + load worklet (once) ─────────────────────────────
async function ensureAudioReady() {
  if (audioReady) return;

  // 1) Create & resume AudioContext
  audioContext = new (window.AudioContext || window.webkitAudioContext)();
  await audioContext.resume();
  playbackRate = audioContext.sampleRate;
  console.log("[DEBUG] audioContext.sampleRate =", playbackRate);

  // 2) Load the playback-processor worklet
  try {
    await audioContext.audioWorklet.addModule("/static/playback-processor.js");
  } catch (err) {
    console.error("[ERROR] Could not load playback-processor.js:", err);
    return;
  }

  // 3) Create the AudioWorkletNode
  playProcessorNode = new AudioWorkletNode(audioContext, "playback-processor");
  playProcessorNode.connect(audioContext.destination);

  console.log("[DEBUG] AudioWorkletNode(playback-processor) created");
  audioReady = true;
}

// ─── Start everything when user clicks “Start the Conversation” ───────────────
async function startStreaming() {
  // 1) Ensure AudioContext + worklet is ready
  await ensureAudioReady();

  // 2) Request microphone (user gesture)
  try {
    micStream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true },
    });
    micSampleRate = audioContext.sampleRate;
    console.log("[DEBUG] Mic sampleRate =", micSampleRate);
  } catch (err) {
    console.error("[UI] getUserMedia error:", err);
    return;
  }

  // 3) Open WebSocket (includes ?lang=selectedLanguage)
  const socket = setupWebSocket();

  // 4) Once WS is open, start mic streaming
  socket.addEventListener("open", () => {
    console.log("[WS] Ready → starting mic streaming");
    startMicStreaming(socket);
  });
}

// ─── Wire Start/Stop button ───────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  const btn = document.getElementById("startStopBtn");
  let streaming = false;

  btn.addEventListener("click", async () => {
    if (!streaming) {
      btn.textContent = "Stop the Conversation";
      streaming = true;
      // Kick off the streaming logic (AudioContext+Worklet+mics+WS)
      await startStreaming();
    } else {
      btn.textContent = "Start the Conversation";
      streaming = false;

      // Tear down mic processing
      if (scriptNodeSender) {
        scriptNodeSender.disconnect();
        scriptNodeSender.onaudioprocess = null;
        scriptNodeSender = null;
      }
      if (micStream) {
        micStream.getTracks().forEach((t) => t.stop());
        micStream = null;
      }
      float32Queue = [];

      // Close WebSocket
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.close();
      }

      // Clear status message + transcripts
      document.getElementById("statusMessage").textContent = "";
      document.getElementById("transcripts").textContent = "";
      playbackEndTime = 0;
    }
  });
});

