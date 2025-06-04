//   let audioContext = null;
//   let playbackRate = null;
//   let playQueue = [];        // FIFO of Float32 bot samples
//   let playerNode = null;     // ScriptProcessor for playback
//   let scriptNodeSender = null;// ScriptProcessor for mic→WS
//   let micStream = null;
//   let float32Queue = [];     // FIFO of Float32 mic samples
//   const TARGET_SAMPLE_RATE = 16000;
//   let micSampleRate = 48000;

//   // Replace the old boolean with these two new variables:
//   let isBotSpeaking = false;         // means “mute mic”
//   let botSilenceTimer = null;        // timeout reference

//   // ─── INITIAL SETUP: AudioContext + playback node ─────────────────────────
//   window.addEventListener('load', async () => {
//     // Create the AudioContext (will resume on user gesture)
//     audioContext = new (window.AudioContext || window.webkitAudioContext)();
//     await audioContext.resume();
//     playbackRate = audioContext.sampleRate;
//     console.log("[DEBUG] playbackRate =", playbackRate);

//     // Create a single ScriptProcessor to continuously pull from playQueue → speakers
//     playerNode = audioContext.createScriptProcessor(4096, 1, 1);
//     playerNode.onaudioprocess = (ev) => {
//       const output = ev.outputBuffer.getChannelData(0);
//       for (let i = 0; i < output.length; i++) {
//         if (playQueue.length > 0) {
//           output[i] = playQueue.shift();
//         } else {
//           output[i] = 0;
//           // We no longer set isBotSpeaking = false here; that was too eager.
//         }
//       }
//     };
//     playerNode.connect(audioContext.destination);
//   });

//   // ─── MICROPHONE CAPTURE (mute whenever isBotSpeaking=true) ────────────────
//   function startMicStreaming(ws) {
//     const bufferSize = 4096;
//     scriptNodeSender = audioContext.createScriptProcessor(bufferSize, 1, 1);
//     const micSource = audioContext.createMediaStreamSource(micStream);
//     micSource.connect(scriptNodeSender);

//     scriptNodeSender.onaudioprocess = (event) => {
//       // If the bot is speaking (or within 300ms since last TTS chunk), skip sending mic audio
//       if (isBotSpeaking) return;

//       const inData = event.inputBuffer.getChannelData(0);
//       float32Queue.push(new Float32Array(inData));
//       let totalSamples = float32Queue.reduce((sum, arr) => sum + arr.length, 0);
//       const needed = Math.ceil((micSampleRate / TARGET_SAMPLE_RATE) * 320);
//       if (totalSamples < needed) return;

//       // Merge all queued floats
//       const merged = new Float32Array(totalSamples);
//       let offset = 0;
//       float32Queue.forEach(chunk => {
//         merged.set(chunk, offset);
//         offset += chunk.length;
//       });

//       // Downsample to 16 kHz
//       const ratio = micSampleRate / TARGET_SAMPLE_RATE;
//       const newLen = Math.floor(merged.length / ratio);
//       const down = new Float32Array(newLen);
//       for (let i = 0; i < newLen; i++) {
//         const start = Math.floor(i * ratio);
//         const end = Math.min(merged.length, Math.floor((i+1) * ratio));
//         let sum = 0, count = 0;
//         for (let j = start; j < end; j++) {
//           sum += merged[j];
//           count++;
//         }
//         down[i] = count > 0 ? (sum / count) : 0;
//       }

//       // Send 320-sample (20ms) blocks to WS as Int16
//       let i = 0;
//       while (i + 320 <= down.length) {
//         const slice = down.subarray(i, i + 320);
//         const int16 = new Int16Array(320);
//         for (let k = 0; k < 320; k++) {
//           const s = Math.max(-1, Math.min(1, slice[k]));
//           int16[k] = (s < 0 ? s * 0x8000 : s * 0x7fff);
//         }
//         if (ws && ws.readyState === WebSocket.OPEN) {
//           ws.send(int16.buffer);
//         }
//         i += 320;
//       }

//       // Keep leftover in float32Queue
//       const leftoverIn = Math.round((down.length - i) * ratio);
//       const leftover = merged.subarray(merged.length - leftoverIn);
//       float32Queue = [leftover];
//     };

//     // Actually connect it so the mic starts firing onaudioprocess
//     scriptNodeSender.connect(audioContext.destination);
//     //   const silentGain = audioContext.createGain();
//     //     silentGain.gain.value = 0;         // “zero out” any audio passing through
//     //     scriptNodeSender.connect(silentGain);
//     //     silentGain.connect(audioContext.destination);
//   }

//   // ─── PLAYBACK: queue incoming TTS chunks + schedule “silence→unmute” ──────
//   function handleBinaryFrame(arrayBuffer) {
//     const pcm16 = new Int16Array(arrayBuffer);
//     const floats48k = new Float32Array(pcm16.length);
//     for (let i = 0; i < pcm16.length; i++) {
//       floats48k[i] = pcm16[i] / 32768;
//     }

//     // Resample only if playbackRate ≠ 48000
//     // if (playbackRate !== 48000) {
//     //   const ratio = 48000 / playbackRate;
//     //   const newLen = Math.round(floats48k.length / ratio);
//     //   const resampled = new Float32Array(newLen);
//     //   for (let i = 0; i < newLen; i++) {
//     //     const idx = i * ratio;
//     //     const i0 = Math.floor(idx);
//     //     const i1 = Math.min(floats48k.length - 1, i0 + 1);
//     //     const w = idx - i0;
//     //     resampled[i] = (1 - w) * floats48k[i0] + w * floats48k[i1];
//     //   }
//     //   // Append to playQueue
//     //   for (let f of resampled) {
//     //     playQueue.push(f);
//     //   }
//     // } else {
//       // Directly queue 48 kHz floats
//       for (let f of floats48k) {
//         playQueue.push(f);
//       }
//     // }

//     // ─── Anytime a new TTS chunk arrives ───────────────────────────────
//     // 1) Immediately mute the mic
//     isBotSpeaking = true;

//     // 2) Clear any previously scheduled “unmute” timer
//     if (botSilenceTimer) {
//       clearTimeout(botSilenceTimer);
//       botSilenceTimer = null;
//     }

//     // 3) Start a fresh 300 ms timeout to “unmute” after no further chunks
//     botSilenceTimer = setTimeout(() => {
//       isBotSpeaking = false;
//       console.log("[DEBUG] isBotSpeaking → false (no new TTS for 100 ms)");
//       botSilenceTimer = null;
//     }, 100);
//   }

//   // ─── WEBSOCKET SETUP: handle JSON + binary frames ─────────────────────
//   function setupWebSocket() {
//     const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
//     const ws = new WebSocket(`${protocol}://${window.location.host}/ws`);
//     ws.binaryType = 'arraybuffer';

//     ws.onopen = () => {
//       console.log('[WS] Connection opened');
//       // We could also clear any stale buffers here:
//       // playQueue = []; isBotSpeaking = false; if (botSilenceTimer) clearTimeout(botSilenceTimer);
//     };
//     ws.onclose = () => console.log('[WS] Connection closed');
//     ws.onerror = err => console.error('[WS] Error:', err);

//     // Flag/logic for displaying transcripts & bot tokens (unchanged)
//     let awaitingBot = false;

//     ws.onmessage = evt => {
//       if (evt.data instanceof ArrayBuffer) {
//         // TTS chunk → queue for playback + schedule “silence” logic
//         console.log('[WS] → Binary chunk, length =', evt.data.byteLength);
//         handleBinaryFrame(evt.data);
//       }
//       else if (evt.data instanceof Blob) {
//         evt.data.arrayBuffer().then(ab => {
//           console.log('[WS] → Blob→ArrayBuffer, length =', ab.byteLength);
//           handleBinaryFrame(ab);
//         });
//       }
//       else {
//         // JSON transcript/token
//         try {
//           const msg = JSON.parse(evt.data);

//           if (msg.type === 'transcript') {
//             const label = msg.final ? 'FINAL' : 'INTERIM';
//             document.getElementById('transcripts').textContent +=
//               `\nTRANSCRIPT [${label}]: ${msg.text}\n`;

//             if (msg.final) {
//               awaitingBot = true;
//             }
//           }
//           else if (msg.type === 'token') {
//             if (awaitingBot) {
//               document.getElementById('transcripts').textContent += "Bot: ";
//               awaitingBot = false;
//             }
//             document.getElementById('transcripts').textContent += msg.token;
//           }
//         }
//         catch (e) {
//           console.warn('[WS] Non‑JSON message:', evt.data);
//         }
//       }
//     };

//     return ws;
//   }

//   // ─── START/STOP LOGIC: connect WebSocket, grab mic, etc. ───────────────
//   async function startStreaming() {
//     if (audioContext.state === 'suspended') {
//       await audioContext.resume();
//       console.log('[DEBUG] audioContext resumed; state =', audioContext.state);
//     }

//     const ws = setupWebSocket();

//     try {
//       micStream = await navigator.mediaDevices.getUserMedia({
//         audio: {
//           echoCancellation: true,
//           noiseSuppression: true
//         }
//       });
//       micSampleRate = audioContext.sampleRate;
//       startMicStreaming(ws);
//     }
//     catch(err) {
//       console.error('[UI] getUserMedia error:', err);
//     }
//   }

//   document.addEventListener('DOMContentLoaded', () => {
//     const btn = document.getElementById('startStopBtn');
//     let streaming = false;

//     btn.addEventListener('click', async () => {
//       if (!streaming) {
//         btn.textContent = 'Stop';
//         streaming = true;
//         await startStreaming();
//       } else {
//         btn.textContent = 'Start';
//         streaming = false;

//         // Tear down mic streamer
//         if (scriptNodeSender) {
//           scriptNodeSender.disconnect();
//           scriptNodeSender.onaudioprocess = null;
//           scriptNodeSender = null;
//         }
//         if (micStream) {
//           micStream.getTracks().forEach(t => t.stop());
//           micStream = null;
//         }
//         float32Queue = [];

//         // Close WS
//         if (ws && ws.readyState === WebSocket.OPEN) {
//           ws.close();
//         }

//         // Clear leftover playback + state
//         playQueue = [];
//         isBotSpeaking = false;
//         if (botSilenceTimer) {
//           clearTimeout(botSilenceTimer);
//           botSilenceTimer = null;
//         }
//       }
//     });
//   });


let audioContext = null;
let playbackRate = null;
let playQueue = [];          // FIFO of Float32 bot samples
let playerNode = null;       // ScriptProcessor for playback
let scriptNodeSender = null; // ScriptProcessor for mic→WS
let micStream = null;
let float32Queue = [];       // FIFO of Float32 mic samples
const TARGET_SAMPLE_RATE = 16000;
let micSampleRate = 48000;
let isBotSpeaking = false;   // flag: bot audio is playing

// Count consecutive empty‐buffer callbacks before unmuting (4 cycles)
let emptyBufferCount = 0;

// ─── On load: create AudioContext + playback node ─────────────────────────────
window.addEventListener('load', async () => {
  audioContext = new (window.AudioContext || window.webkitAudioContext)();
  await audioContext.resume();
  playbackRate = audioContext.sampleRate;      // typically 48000
  console.log("[DEBUG] playbackRate =", playbackRate);

  // Playback node: drains playQueue; only unmute after 4 empty cycles (~340 ms)
  playerNode = audioContext.createScriptProcessor(4096, 1, 1);
  playerNode.onaudioprocess = (ev) => {
    const output = ev.outputBuffer.getChannelData(0);
    if (playQueue.length > 0) {
      // Drain up to 4096 samples
      for (let i = 0; i < output.length; i++) {
        output[i] = playQueue.length > 0 ? playQueue.shift() : 0;
      }
      emptyBufferCount = 0;
    } else {
      // No data: output silence
      for (let i = 0; i < output.length; i++) {
        output[i] = 0;
      }
      if (isBotSpeaking) {
        emptyBufferCount++;
        if (emptyBufferCount >= 4) {
          isBotSpeaking = false;
          emptyBufferCount = 0;
          console.log("[DEBUG] isBotSpeaking → false (4 empty cycles)");
        }
      }
    }
  };
  playerNode.connect(audioContext.destination);
});

// ─── Downsample Float32Array [srcRate→16 kHz] ──────────────────────────────────
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

// ─── Convert Float32Array [–1..1] → Int16Array ─────────────────────────────────
function floatToInt16(floatBuffer) {
  const int16 = new Int16Array(floatBuffer.length);
  for (let i = 0; i < floatBuffer.length; i++) {
    const s = Math.max(-1, Math.min(1, floatBuffer[i]));
    int16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return int16;
}

// ─── Capture mic→WS, skipping when isBotSpeaking=true ───────────────────────────
function startMicStreaming(ws) {
  const bufferSize = 4096;
  scriptNodeSender = audioContext.createScriptProcessor(bufferSize, 1, 1);
  const micSource = audioContext.createMediaStreamSource(micStream);
  micSource.connect(scriptNodeSender);

  scriptNodeSender.onaudioprocess = (event) => {
    if (isBotSpeaking) return;

    const inData = event.inputBuffer.getChannelData(0);
    float32Queue.push(new Float32Array(inData));

    let totalSamples = float32Queue.reduce((sum, arr) => sum + arr.length, 0);
    const needed = Math.ceil((micSampleRate / TARGET_SAMPLE_RATE) * 320);
    if (totalSamples < needed) return;

    // Merge queued floats
    const merged = new Float32Array(totalSamples);
    let offset = 0;
    float32Queue.forEach(chunk => {
      merged.set(chunk, offset);
      offset += chunk.length;
    });

    // Downsample to 16 kHz
    const ratio = micSampleRate / TARGET_SAMPLE_RATE;
    const newLen = Math.floor(merged.length / ratio);
    const down = new Float32Array(newLen);
    for (let i = 0; i < newLen; i++) {
      const start = Math.floor(i * ratio);
      const end = Math.min(merged.length, Math.floor((i + 1) * ratio));
      let sum = 0, count = 0;
      for (let j = start; j < end; j++) {
        sum += merged[j];
        count++;
      }
      down[i] = count > 0 ? (sum / count) : 0;
    }

    // Send 320-sample (20 ms) frames as Int16
    let i = 0;
    while (i + 320 <= down.length) {
      const slice = down.subarray(i, i + 320);
      const int16 = new Int16Array(320);
      for (let k = 0; k < 320; k++) {
        const s = Math.max(-1, Math.min(1, slice[k]));
        int16[k] = s < 0 ? s * 0x8000 : s * 0x7fff;
      }
      if (ws.readyState === WebSocket.OPEN) {
        ws.send(int16.buffer);
        console.debug('[MIC] Sent 320 samples to server');
      }
      i += 320;
    }

    // Keep leftover samples
    const leftoverIn = Math.round((down.length - i) * ratio);
    const leftover = merged.subarray(merged.length - leftoverIn);
    float32Queue = [leftover];
  };

  scriptNodeSender.connect(audioContext.destination);
  console.log('[MIC] Mic streaming started');
}

// ─── Handle incoming TTS chunks: convert/resample/enqueue ──────────────────────
function handleBinaryFrame(arrayBuffer) {
  const pcm16 = new Int16Array(arrayBuffer);
  if (!pcm16.length) return;

  const floats48k = new Float32Array(pcm16.length);
  for (let i = 0; i < pcm16.length; i++) {
    floats48k[i] = pcm16[i] / 32768;
  }

  // Resample to playbackRate if ≠ 48000
  if (playbackRate !== 48000) {
    const ratio = 48000 / playbackRate;
    const newLen = Math.round(floats48k.length / ratio);
    const resampled = new Float32Array(newLen);
    for (let i = 0; i < newLen; i++) {
      const idx = i * ratio;
      const i0 = Math.floor(idx);
      const i1 = Math.min(floats48k.length - 1, i0 + 1);
      const w = idx - i0;
      resampled[i] = (1 - w) * floats48k[i0] + w * floats48k[i1];
    }
    playQueue.push(...resampled);
  } else {
    playQueue.push(...floats48k);
  }

  // Mark bot speaking and reset emptyBufferCount
  isBotSpeaking = true;
  emptyBufferCount = 0;
  console.log("[DEBUG] isBotSpeaking → true (new chunk)");
}

// ─── Build WebSocket and attach handlers ───────────────────────────────────────
function setupWebSocket() {
  const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
  ws = new WebSocket(`${protocol}://${window.location.host}/ws`);
  ws.binaryType = 'arraybuffer';

  ws.onopen = () => console.log('[WS] Connection opened');
  ws.onclose = () => console.log('[WS] Connection closed');
  ws.onerror = (err) => console.error('[WS] Error:', err);

  ws.onmessage = (evt) => {
    if (evt.data instanceof ArrayBuffer) {
      console.log('[WS] → Binary chunk length =', evt.data.byteLength);
      handleBinaryFrame(evt.data);
    } else if (evt.data instanceof Blob) {
      evt.data.arrayBuffer().then(ab => {
        console.log('[WS] → Blob→ArrayBuffer length =', ab.byteLength);
        handleBinaryFrame(ab);
      }).catch(err => console.error('[WS] Blob→ArrayBuffer error:', err));
    } else {
      try {
        const msg = JSON.parse(evt.data);
        if (msg.type === 'transcript') {
          const label = msg.final ? 'FINAL' : 'INTERIM';
          document.getElementById('transcripts').textContent +=
            `TRANSCRIPT [${label}]: ${msg.text}\n`;
        } else if (msg.type === 'token') {
          document.getElementById('transcripts').textContent += msg.token;
        }
      } catch (e) {
        console.warn('[WS] Non-JSON message:', evt.data);
      }
    }
  };

  return ws;
}

// ─── Start capturing mic → WS and enable playback ─────────────────────────────
async function startStreaming() {
  // 1) Resume AudioContext if needed
  if (audioContext.state === 'suspended') {
    await audioContext.resume();
    console.log('[DEBUG] audioContext resumed; state =', audioContext.state);
  }

  // 2) Request mic
  try {
    micStream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true }
    });
    micSampleRate = audioContext.sampleRate;
    console.log('[DEBUG] Mic sampleRate =', micSampleRate);
  } catch(err) {
    console.error('[UI] getUserMedia error:', err);
    return;
  }

  // 3) Open WebSocket
  const socket = setupWebSocket();

  // 4) Once WS is open, start mic streaming
  socket.addEventListener('open', () => {
    console.log('[WS] Ready → starting mic streaming');
    startMicStreaming(socket);
  });
}

// ─── Wire Start/Stop button ───────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  const btn = document.getElementById('startStopBtn');
  let streaming = false;

  btn.addEventListener('click', async () => {
    if (!streaming) {
      btn.textContent = 'Stop';
      streaming = true;
      await startStreaming();
    } else {
      btn.textContent = 'Start';
      streaming = false;

      // Tear down mic capture
      if (scriptNodeSender) {
        scriptNodeSender.disconnect();
        scriptNodeSender.onaudioprocess = null;
        scriptNodeSender = null;
      }
      if (micStream) {
        micStream.getTracks().forEach(t => t.stop());
        micStream = null;
      }
      float32Queue = [];

      // Close WebSocket
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.close();
      }

      // Clear playback queue and reset state
      playQueue = [];
      isBotSpeaking = false;
      emptyBufferCount = 0;
    }
  });
});


// static/app.js

// let audioContext = null;
// let playbackRate = null;
// let playQueue = [];         // FIFO of Float32 bot audio samples
// let playerNode = null;      // ScriptProcessor for playback
// let scriptNodeSender = null;// ScriptProcessor for mic→WS
// let micStream = null;
// let float32Queue = [];      // FIFO of Float32 mic samples
// const TARGET_SAMPLE_RATE = 16000;
// let micSampleRate = 48000;

// // Flags and timer to keep mic muted while bot is speaking
// let isBotSpeaking = false;
// let botSilenceTimer = null;

// window.addEventListener('load', async () => {
//   // 1) Create AudioContext and resume (so sampleRate is accurate)
//   audioContext = new (window.AudioContext || window.webkitAudioContext)();
//   await audioContext.resume();
//   playbackRate = audioContext.sampleRate;
//   console.log('[DEBUG] playbackRate =', playbackRate);

//   // 2) Build a single ScriptProcessorNode to continuously drain playQueue → speakers
//   playerNode = audioContext.createScriptProcessor(4096, 1, 1);
//   playerNode.onaudioprocess = (ev) => {
//     const output = ev.outputBuffer.getChannelData(0);
//     for (let i = 0; i < output.length; i++) {
//       if (playQueue.length > 0) {
//         output[i] = playQueue.shift();
//       } else {
//         output[i] = 0;
//         // We no longer set isBotSpeaking = false here; that logic is handled by our silence timer
//       }
//     }
//   };
//   playerNode.connect(audioContext.destination);
// });

// // ─── MICROPHONE CAPTURE (keep mic muted when isBotSpeaking=true) ──────────
// function startMicStreaming(ws) {
//   const bufferSize = 4096;
//   scriptNodeSender = audioContext.createScriptProcessor(bufferSize, 1, 1);
//   const micSource = audioContext.createMediaStreamSource(micStream);
//   micSource.connect(scriptNodeSender);

//   scriptNodeSender.onaudioprocess = (event) => {
//     // If the bot is currently speaking (or within the silence‐timeout window), skip sending mic frames
//     if (isBotSpeaking) return;

//     const inData = event.inputBuffer.getChannelData(0);
//     float32Queue.push(new Float32Array(inData));
//     let totalSamples = float32Queue.reduce((sum, arr) => sum + arr.length, 0);
//     const needed = Math.ceil((micSampleRate / TARGET_SAMPLE_RATE) * 320);
//     if (totalSamples < needed) return;

//     // Merge queued floats
//     const merged = new Float32Array(totalSamples);
//     let offset = 0;
//     for (const chunk of float32Queue) {
//       merged.set(chunk, offset);
//       offset += chunk.length;
//     }

//     // Downsample from micSampleRate → TARGET_SAMPLE_RATE (16kHz)
//     const ratio = micSampleRate / TARGET_SAMPLE_RATE;
//     const newLen = Math.floor(merged.length / ratio);
//     const down = new Float32Array(newLen);
//     for (let i = 0; i < newLen; i++) {
//       const start = Math.floor(i * ratio);
//       const end = Math.min(merged.length, Math.floor((i + 1) * ratio));
//       let sum = 0, count = 0;
//       for (let j = start; j < end; j++) {
//         sum += merged[j];
//         count++;
//       }
//       down[i] = count > 0 ? (sum / count) : 0;
//     }

//     // Send 320‑sample slices (20 ms at 16kHz) as 16‑bit PCM to the server
//     let i = 0;
//     while (i + 320 <= down.length) {
//       const slice = down.subarray(i, i + 320);
//       const int16 = new Int16Array(320);
//       for (let k = 0; k < 320; k++) {
//         const s = Math.max(-1, Math.min(1, slice[k]));
//         int16[k] = (s < 0 ? s * 0x8000 : s * 0x7fff);
//       }
//       if (ws && ws.readyState === WebSocket.OPEN) {
//         ws.send(int16.buffer);
//       }
//       i += 320;
//     }

//     // Keep leftover in float32Queue
//     const leftoverIn = Math.round((down.length - i) * ratio);
//     const leftover = merged.subarray(merged.length - leftoverIn);
//     float32Queue = [leftover];
//   };

//   // To keep the ScriptProcessor “alive,” route it through a zero‑gain node:
//   const silentGain = audioContext.createGain();
//   silentGain.gain.value = 0;               // completely mute this path
//   scriptNodeSender.connect(silentGain);
//   silentGain.connect(audioContext.destination);
// }

// // ─── PLAYBACK: queue incoming TTS chunks + schedule “mute/unmute” ─────────
// function handleBinaryFrame(arrayBuffer) {
//   const pcm16 = new Int16Array(arrayBuffer);
//   const floats48k = new Float32Array(pcm16.length);
//   for (let i = 0; i < pcm16.length; i++) {
//     floats48k[i] = pcm16[i] / 32768;
//   }

//   // Resample 48 kHz → playbackRate if needed
//   if (playbackRate !== 48000) {
//     const ratio = 48000 / playbackRate;
//     const newLen = Math.round(floats48k.length / ratio);
//     const resampled = new Float32Array(newLen);
//     for (let i = 0; i < newLen; i++) {
//       const idx = i * ratio;
//       const i0 = Math.floor(idx);
//       const i1 = Math.min(floats48k.length - 1, i0 + 1);
//       const w = idx - i0;
//       resampled[i] = (1 - w) * floats48k[i0] + w * floats48k[i1];
//     }
//     for (const f of resampled) {
//       playQueue.push(f);
//     }
//   } else {
//     for (const f of floats48k) {
//       playQueue.push(f);
//     }
//   }

//   // Mute mic immediately as soon as any TTS chunk arrives
//   isBotSpeaking = true;

//   // Clear any existing unmute timer
//   if (botSilenceTimer) {
//     clearTimeout(botSilenceTimer);
//     botSilenceTimer = null;
//   }

//   // Start (or restart) an 800 ms timer; only after no TTS for 800 ms will mic unmute
//   botSilenceTimer = setTimeout(() => {
//     isBotSpeaking = false;
//     console.log('[DEBUG] isBotSpeaking → false (no new TTS for 200 ms)');
//     botSilenceTimer = null;
//   }, 200);
// }

// // ─── WEBSOCKET SETUP: handle both binary (audio) and JSON (transcripts/tokens) ─
// function setupWebSocket() {
//   const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
//   const ws = new WebSocket(`${protocol}://${window.location.host}/ws`);
//   ws.binaryType = 'arraybuffer';

//   ws.onopen = () => {
//     console.log('[WS] Connection opened');
//     // You could clear leftover state here if desired:
//     // playQueue = []; isBotSpeaking = false; if (botSilenceTimer) clearTimeout(botSilenceTimer);
//   };
//   ws.onclose = () => console.log('[WS] Connection closed');
//   ws.onerror = (err) => console.error('[WS] Error:', err);

//   // Flag to prefix “Bot: ” on the very first token after a final transcript
//   let awaitingBot = false;

//   ws.onmessage = (evt) => {
//     if (evt.data instanceof ArrayBuffer) {
//       // Incoming TTS chunk → queue for playback + manage silence timer
//       console.log('[WS] → Binary chunk, length =', evt.data.byteLength);
//       handleBinaryFrame(evt.data);
//     }
//     else if (evt.data instanceof Blob) {
//       // Some browsers wrap binary as Blob; convert to ArrayBuffer first
//       evt.data.arrayBuffer().then((ab) => {
//         console.log('[WS] → Blob→ArrayBuffer, length =', ab.byteLength);
//         handleBinaryFrame(ab);
//       });
//     }
//     else {
//       // JSON‐formatted message (transcript or token)
//       try {
//         const msg = JSON.parse(evt.data);

//         if (msg.type === 'transcript') {
//           const label = msg.final ? 'FINAL' : 'INTERIM';
//           // Start each transcript on its own line and end with a newline
//           document.getElementById('transcripts').textContent +=
//             `\nTRANSCRIPT [${label}]: ${msg.text}\n`;

//           if (msg.final) {
//             // Next incoming token(s) belong to the bot
//             awaitingBot = true;
//           }
//         }
//         else if (msg.type === 'token') {
//           if (awaitingBot) {
//             // Prefix “Bot: ” once, then print tokens continuously
//             document.getElementById('transcripts').textContent += 'Bot: ';
//             awaitingBot = false;
//           }
//           document.getElementById('transcripts').textContent += msg.token;
//         }
//         else if (msg.type === 'response_end') {
//           // (Optional) You could clear or log something here if needed
//           console.log('[WS] Received response_end (LLM finished streaming tokens)');
//         }
//       }
//       catch (e) {
//         console.warn('[WS] Non‑JSON message:', evt.data);
//       }
//     }
//   };

//   return ws;
// }

// // ─── START/STOP LOGIC: open WS, resume AudioContext, grab mic ────────────
// async function startStreaming() {
//   // Resume AudioContext on user gesture
//   if (audioContext.state === 'suspended') {
//     await audioContext.resume();
//     console.log('[DEBUG] audioContext resumed; state =', audioContext.state);
//   }

//   // Open WebSocket and begin playback/JSON handling
//   const ws = setupWebSocket();

//   // Grab microphone (with echo cancellation + noise suppression), then start sending
//   try {
//     micStream = await navigator.mediaDevices.getUserMedia({
//       audio: {
//         echoCancellation: true,
//         noiseSuppression: true
//       }
//     });
//     micSampleRate = audioContext.sampleRate;
//     startMicStreaming(ws);
//   } 
//   catch (err) {
//     console.error('[UI] getUserMedia error:', err);
//   }
// }

// // When the user clicks Start/Stop, toggle streaming
// document.addEventListener('DOMContentLoaded', () => {
//   const btn = document.getElementById('startStopBtn');
//   let streaming = false;
//   let wsRef = null;

//   btn.addEventListener('click', async () => {
//     if (!streaming) {
//       btn.textContent = 'Stop';
//       streaming = true;
//       wsRef = await startStreaming();
//     } else {
//       btn.textContent = 'Start';
//       streaming = false;

//       // Tear down mic stream
//       if (scriptNodeSender) {
//         scriptNodeSender.disconnect();
//         scriptNodeSender.onaudioprocess = null;
//         scriptNodeSender = null;
//       }
//       if (micStream) {
//         micStream.getTracks().forEach(t => t.stop());
//         micStream = null;
//       }
//       float32Queue = [];

//       // Close WebSocket
//       if (wsRef && wsRef.readyState === WebSocket.OPEN) {
//         wsRef.close();
//       }
//       wsRef = null;

//       // Clear playback queue and reset flags
//       playQueue = [];
//       isBotSpeaking = false;
//       if (botSilenceTimer) {
//         clearTimeout(botSilenceTimer);
//         botSilenceTimer = null;
//       }
//     }
//   });
// });



// static/app.js

// let audioContext = null;
// let micProcessor = null;      // ScriptProcessorNode for mic capture
// let playerProcessor = null;   // ScriptProcessorNode for playback
// let micStream = null;
// let ws = null;
// let playQueue = [];           // FIFO of Float32 samples for playback
// const TARGET_SAMPLE_RATE = 16000;

// // Flags and timer to keep mic muted while bot is speaking
// let isBotSpeaking = false;
// let botSilenceTimer = null;

// // Helper: Downsample Float32Array from srcRate → TARGET_SAMPLE_RATE
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

// // Convert Float32Array [-1,1] to Int16Array
// function floatToInt16(floatBuffer) {
//   const int16 = new Int16Array(floatBuffer.length);
//   for (let i = 0; i < floatBuffer.length; i++) {
//     const s = Math.max(-1, Math.min(1, floatBuffer[i]));
//     int16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
//   }
//   return int16;
// }

// // Convert raw 16-bit PCM (ArrayBuffer) at 48000 Hz → Float32Array [-1,1]
// function int16ToFloat32(buffer) {
//   const int16 = new Int16Array(buffer);
//   const float32 = new Float32Array(int16.length);
//   for (let i = 0; i < int16.length; i++) {
//     float32[i] = int16[i] / 32768;
//   }
//   return float32;
// }

// // Set up WebSocket, audio capture, and playback
// async function startStreaming() {
//   // 1) Create/resume AudioContext
//   if (!audioContext) {
//     audioContext = new (window.AudioContext || window.webkitAudioContext)();
//   }
//   if (audioContext.state === 'suspended') {
//     await audioContext.resume();
//   }

//   // 2) Open WebSocket
//   const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
//   ws = new WebSocket(`${protocol}://${window.location.host}/ws`);
//   ws.binaryType = 'arraybuffer';

//   ws.onopen = () => {
//     console.log('[WS] Connection opened');
//   };
//   ws.onclose = () => console.log('[WS] Connection closed');
//   ws.onerror = (err) => console.error('[WS] Error:', err);

//   // 3) Handle incoming WebSocket messages
//   let awaitingBot = false;
//   ws.onmessage = (evt) => {
//     if (evt.data instanceof ArrayBuffer) {
//       // TTS PCM chunk at 48000 Hz, Int16
//       const floats = int16ToFloat32(evt.data);
//       const playbackRate = audioContext.sampleRate;
//       let toQueue;
//       if (playbackRate !== 48000) {
//         const ratio = 48000 / playbackRate;
//         const newLen = Math.round(floats.length / ratio);
//         const resampled = new Float32Array(newLen);
//         for (let i = 0; i < newLen; i++) {
//           const idx = i * ratio;
//           const i0 = Math.floor(idx);
//           const i1 = Math.min(floats.length - 1, i0 + 1);
//           const w = idx - i0;
//           resampled[i] = (1 - w) * floats[i0] + w * floats[i1];
//         }
//         toQueue = resampled;
//       } else {
//         toQueue = floats;
//       }
//       for (const s of toQueue) playQueue.push(s);

//       // Mute mic immediately as soon as any TTS chunk arrives
//       isBotSpeaking = true;
//       if (botSilenceTimer) {
//         clearTimeout(botSilenceTimer);
//         botSilenceTimer = null;
//       }
//       // Use 800ms timeout before unmuting
//       botSilenceTimer = setTimeout(() => {
//         isBotSpeaking = false;
//         console.log('[DEBUG] isBotSpeaking → false (no new TTS for 300ms)');
//         botSilenceTimer = null;
//       }, 300);
//     }
//     else if (evt.data instanceof Blob) {
//       evt.data.arrayBuffer().then((ab) => {
//         const floats = int16ToFloat32(ab);
//         const playbackRate = audioContext.sampleRate;
//         let toQueue;
//         if (playbackRate !== 48000) {
//           const ratio = 48000 / playbackRate;
//           const newLen = Math.round(floats.length / ratio);
//           const resampled = new Float32Array(newLen);
//           for (let i = 0; i < newLen; i++) {
//             const idx = i * ratio;
//             const i0 = Math.floor(idx);
//             const i1 = Math.min(floats.length - 1, i0 + 1);
//             const w = idx - i0;
//             resampled[i] = (1 - w) * floats[i0] + w * floats[i1];
//           }
//           toQueue = resampled;
//         } else {
//           toQueue = floats;
//         }
//         for (const s of toQueue) playQueue.push(s);

//         isBotSpeaking = true;
//         if (botSilenceTimer) {
//           clearTimeout(botSilenceTimer);
//           botSilenceTimer = null;
//         }
//         botSilenceTimer = setTimeout(() => {
//           isBotSpeaking = false;
//           console.log('[DEBUG] isBotSpeaking → false (no new TTS for 800ms)');
//           botSilenceTimer = null;
//         }, 800);
//       });
//     }
//     else {
//       // JSON: transcript or token
//       try {
//         const msg = JSON.parse(evt.data);
//         if (msg.type === 'transcript') {
//           const label = msg.final ? 'FINAL' : 'INTERIM';
//           document.getElementById('transcripts').textContent +=
//             `\nTRANSCRIPT [${label}]: ${msg.text}\n`;
//           if (msg.final) awaitingBot = true;
//         }
//         else if (msg.type === 'token') {
//           if (awaitingBot) {
//             document.getElementById('transcripts').textContent += 'Bot: ';
//             awaitingBot = false;
//           }
//           document.getElementById('transcripts').textContent += msg.token;
//         }
//       } catch (e) {
//         console.warn('[WS] Non-JSON message:', evt.data);
//       }
//     }
//   };

//   // 4) Set up microphone capture
//   try {
//     micStream = await navigator.mediaDevices.getUserMedia({
//       audio: {
//         echoCancellation: true,
//         noiseSuppression: true
//       }
//     });
//   } catch (err) {
//     console.error('[UI] getUserMedia error:', err);
//     return;
//   }

//   const source = audioContext.createMediaStreamSource(micStream);

//   micProcessor = audioContext.createScriptProcessor(4096, 1, 1);
//   const micSampleRate = audioContext.sampleRate;

//   source.connect(micProcessor);

//   micProcessor.onaudioprocess = (evt) => {
//     if (isBotSpeaking) return;
//     const inData = evt.inputBuffer.getChannelData(0);
//     const down = downsampleBuffer(inData, micSampleRate);
//     const int16 = floatToInt16(down);
//     if (ws && ws.readyState === WebSocket.OPEN) {
//       ws.send(int16.buffer);
//     }
//   };

//   // Keep micProcessor alive but muted via a zero‑gain node
//   const silentGain = audioContext.createGain();
//   silentGain.gain.value = 0;
//   micProcessor.connect(silentGain);
//   silentGain.connect(audioContext.destination);

//   // 5) Set up playback processor to drain playQueue → speakers
//   playerProcessor = audioContext.createScriptProcessor(4096, 1, 1);
//   playerProcessor.onaudioprocess = (evt) => {
//     const out = evt.outputBuffer.getChannelData(0);
//     for (let i = 0; i < out.length; i++) {
//       out[i] = playQueue.length ? playQueue.shift() : 0;
//     }
//   };
//   playerProcessor.connect(audioContext.destination);
// }

// function stopStreaming() {
//   // Tear down mic capture
//   if (micProcessor) {
//     micProcessor.disconnect();
//     micProcessor.onaudioprocess = null;
//     micProcessor = null;
//   }
//   if (micStream) {
//     micStream.getTracks().forEach(t => t.stop());
//     micStream = null;
//   }

//   // Close WebSocket
//   if (ws && ws.readyState === WebSocket.OPEN) {
//     ws.close();
//   }
//   ws = null;

//   // Tear down playback
//   if (playerProcessor) {
//     playerProcessor.disconnect();
//     playerProcessor.onaudioprocess = null;
//     playerProcessor = null;
//   }

//   // Clear queue and transcripts
//   playQueue = [];
//   document.getElementById('transcripts').textContent = '';
// }

// document.addEventListener('DOMContentLoaded', () => {
//   const btn = document.getElementById('startStopBtn');
//   let streaming = false;

//   btn.addEventListener('click', async () => {
//     if (!streaming) {
//       btn.textContent = 'Stop';
//       streaming = true;
//       await startStreaming();
//     } else {
//       btn.textContent = 'Start';
//       streaming = false;
//       stopStreaming();
//     }
//   });
// });



// app.js

// let audioContext = null;
// let micProcessor = null;    // ScriptProcessorNode for mic capture
// let micStream = null;
// let ws = null;

// const TARGET_SAMPLE_RATE = 16000; // 16 kHz for Deepgram

// // Flag and timer to mute mic while bot is speaking
// let isBotSpeaking = false;
// let botSilenceTimer = null;

// // ────────────────────────────────────────────────────────────────────────────────
// // Helper: Downsample Float32Array from srcRate → TARGET_SAMPLE_RATE
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

// // Convert Float32Array [−1..1] to Int16Array
// function floatToInt16(floatBuffer) {
//   const int16 = new Int16Array(floatBuffer.length);
//   for (let i = 0; i < floatBuffer.length; i++) {
//     const s = Math.max(-1, Math.min(1, floatBuffer[i]));
//     int16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
//   }
//   return int16;
// }

// // Convert raw 16-bit PCM (ArrayBuffer) at 48000 Hz → Float32Array [−1..1]
// function int16ToFloat32(buffer) {
//   const int16 = new Int16Array(buffer);
//   const float32 = new Float32Array(int16.length);
//   for (let i = 0; i < int16.length; i++) {
//     float32[i] = int16[i] / 32768;
//   }
//   return float32;
// }

// // ────────────────────────────────────────────────────────────────────────────────
// // Schedule each incoming TTS chunk as its own AudioBufferSource
// function scheduleTTSChunk(arrayBuffer) {
//   if (!audioContext) return;

//   // 1) Convert raw int16 → Float32 [−1..1]
//   const int16 = new Int16Array(arrayBuffer);
//   const float32Raw = new Float32Array(int16.length);
//   for (let i = 0; i < int16.length; i++) {
//     float32Raw[i] = int16[i] / 32768;
//   }

//   // 2) Resample from 48000 → audioContext.sampleRate if necessary
//   const targetRate = audioContext.sampleRate;
//   let floats;
//   if (targetRate !== 48000) {
//     const ratio = 48000 / targetRate;
//     const newLen = Math.round(float32Raw.length / ratio);
//     floats = new Float32Array(newLen);
//     for (let i = 0; i < newLen; i++) {
//       const idx = i * ratio;
//       const i0 = Math.floor(idx);
//       const i1 = Math.min(float32Raw.length - 1, i0 + 1);
//       const w = idx - i0;
//       floats[i] = (1 - w) * float32Raw[i0] + w * float32Raw[i1];
//     }
//   } else {
//     floats = float32Raw;
//   }

//   // 3) Create an AudioBuffer at the correct sampleRate
//   const buf = audioContext.createBuffer(1, floats.length, targetRate);
//   buf.copyToChannel(floats, 0, 0);

//   // 4) Create a BufferSource and play immediately
//   const src = audioContext.createBufferSource();
//   src.buffer = buf;
//   src.connect(audioContext.destination);
//   if (audioContext.state === 'suspended') {
//     audioContext.resume().then(() => src.start());
//   } else {
//     src.start();
//   }
// }

// // ────────────────────────────────────────────────────────────────────────────────
// // Start capturing mic, opening WS, and handling playback/transcripts
// async function startStreaming() {
//   // 1) Create or resume AudioContext
//   if (!audioContext) {
//     audioContext = new (window.AudioContext || window.webkitAudioContext)();
//   }
//   if (audioContext.state === 'suspended') {
//     await audioContext.resume();
//   }

//   // 2) Open WebSocket
//   const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
//   ws = new WebSocket(`${protocol}://${window.location.host}/ws`);
//   ws.binaryType = 'arraybuffer';

//   ws.onopen = () => console.log('[WS] Connection opened');
//   ws.onclose = () => console.log('[WS] Connection closed');
//   ws.onerror = (err) => console.error('[WS] Error:', err);

//   // 3) Handle incoming WS messages
//   let awaitingBot = false;
//   ws.onmessage = (evt) => {
//     // (A) If it’s an ArrayBuffer, treat as TTS PCM @ 48000 Hz
//     if (evt.data instanceof ArrayBuffer) {
//       scheduleTTSChunk(evt.data);

//       // Mute mic while bot speaking; reset timer
//       isBotSpeaking = true;
//       if (botSilenceTimer) {
//         clearTimeout(botSilenceTimer);
//       }
//       botSilenceTimer = setTimeout(() => {
//         isBotSpeaking = false;
//         console.log('[DEBUG] isBotSpeaking → false (no new TTS for 300ms)');
//         botSilenceTimer = null;
//       }, 300);
//     }
//     // (B) If it’s a Blob, convert first, then schedule
//     else if (evt.data instanceof Blob) {
//       evt.data.arrayBuffer().then((ab) => {
//         scheduleTTSChunk(ab);

//         isBotSpeaking = true;
//         if (botSilenceTimer) {
//           clearTimeout(botSilenceTimer);
//         }
//         botSilenceTimer = setTimeout(() => {
//           isBotSpeaking = false;
//           console.log('[DEBUG] isBotSpeaking → false (no new TTS for 800ms)');
//           botSilenceTimer = null;
//         }, 800);
//       });
//     }
//     // (C) Otherwise, treat as JSON (transcript or token)
//     else {
//       try {
//         const msg = JSON.parse(evt.data);
//         if (msg.type === 'transcript') {
//           const label = msg.final ? 'FINAL' : 'INTERIM';
//           document.getElementById('transcripts').textContent +=
//             `\nTRANSCRIPT [${label}]: ${msg.text}\n`;
//           if (msg.final) awaitingBot = true;
//         } else if (msg.type === 'token') {
//           if (awaitingBot) {
//             document.getElementById('transcripts').textContent += 'Bot: ';
//             awaitingBot = false;
//           }
//           document.getElementById('transcripts').textContent += msg.token;
//         }
//       } catch (e) {
//         console.warn('[WS] Non-JSON message:', evt.data);
//       }
//     }
//   };

//   // 4) Request microphone access
//   try {
//     micStream = await navigator.mediaDevices.getUserMedia({
//       audio: { echoCancellation: true, noiseSuppression: true }
//     });
//   } catch (err) {
//     console.error('[UI] getUserMedia error:', err);
//     return;
//   }

//   // 5) Create MediaStreamSource + ScriptProcessorNode for mic
//   const source = audioContext.createMediaStreamSource(micStream);
//   micProcessor = audioContext.createScriptProcessor(4096, 1, 1);
//   const micSampleRate = audioContext.sampleRate;

//   source.connect(micProcessor);
//   micProcessor.onaudioprocess = (evt) => {
//     if (isBotSpeaking) return; // mute mic while bot speaks

//     const inData = evt.inputBuffer.getChannelData(0); // Float32 @ micSampleRate
//     const down = downsampleBuffer(inData, micSampleRate); // → Float32 @ 16000
//     const int16 = floatToInt16(down);                  // → Int16Array

//     if (ws && ws.readyState === WebSocket.OPEN) {
//       ws.send(int16.buffer); // send raw PCM16 to server
//     }
//   };

//   // Keep micProcessor alive but silently connected
//   const silentGain = audioContext.createGain();
//   silentGain.gain.value = 0;
//   micProcessor.connect(silentGain);
//   silentGain.connect(audioContext.destination);
// }

// // ────────────────────────────────────────────────────────────────────────────────
// // Stop everything: mic, WS, playback scheduling
// function stopStreaming() {
//   // Disconnect mic processor
//   if (micProcessor) {
//     micProcessor.disconnect();
//     micProcessor.onaudioprocess = null;
//     micProcessor = null;
//   }
//   if (micStream) {
//     micStream.getTracks().forEach((t) => t.stop());
//     micStream = null;
//   }

//   // Close WebSocket
//   if (ws && ws.readyState === WebSocket.OPEN) {
//     ws.close();
//   }
//   ws = null;

//   // Clear transcripts area
//   document.getElementById('transcripts').textContent = '';
// }

// // ────────────────────────────────────────────────────────────────────────────────
// // Button logic: toggle Start/Stop
// document.addEventListener('DOMContentLoaded', () => {
//   const btn = document.getElementById('startStopBtn');
//   let streaming = false;

//   btn.addEventListener('click', async () => {
//     if (!streaming) {
//       btn.textContent = 'Stop';
//       streaming = true;
//       await startStreaming();
//     } else {
//       btn.textContent = 'Start';
//       streaming = false;
//       stopStreaming();
//     }
//   });
// });
