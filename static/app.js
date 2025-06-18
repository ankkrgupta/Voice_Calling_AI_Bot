// ───────────────────────────────────────────────────────────────────────────────
//                                   app.js                                  
// ───────────────────────────────────────────────────────────────────────────────
//
// Using an AudioWorkletNode for TTS playback and a ScriptProcessorNode for mic capture.
// User can pick “en-IN” or “Hindi” *before* clicking “Start the Conversation.” That
// choice is sent as ?lang=<…> to the WebSocket URL. Once “Start” is clicked, an
// informational message appears showing which language is in effect and how to change it.
// ───────────────────────────────────────────────────────────────────────────────

let audioContext = null;
let playProcessorNode = null;
let captureCtx = null;
let scriptNodeSender = null;
let micStream = null;
let ws = null;

let float32Queue = [];                // FIFO of Float32 mic samples
const TARGET_SAMPLE_RATE = 16000;
const TTS_SAMPLE_RATE    = 44100;
let micSampleRate = null;

let playbackEndTime = 0;
let playbackRate = null;
let selectedLanguage = "en-IN";       // default lang

window.addEventListener("load", () => {
  // Language buttons
  document.querySelectorAll(".langBtn").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".langBtn").forEach(b => b.classList.remove("selected"));
      btn.classList.add("selected");
      selectedLanguage = btn.dataset.lang;
      console.log(`[UI] Selected language = ${selectedLanguage}`);
    });
  });

  // Start/Stop button
  const startStopBtn = document.getElementById("startStopBtn");
  let streaming = false;

  startStopBtn.addEventListener("click", async () => {
    if (!streaming) {
      // ─── USER CLICKED “START” ───────────────────────────────────────
      streaming = true;
      startStopBtn.textContent = "Stop the Conversation";
      startStopBtn.classList.add("stop");
      document.getElementById("info").textContent =
        selectedLanguage === "en-IN"
          ? "Current language of conversation is English. " + "If you want to shift to Hindi, select it above, then stop the ongoing conversation and click on start conversation button again."
          : "Current language of conversation is Hindi. " + "If you want to shift to English, select it above, then stop the ongoing conversation and click on start conversation button again.";

      // 1) Create & resume playback AudioContext
      audioContext = new (window.AudioContext || window.webkitAudioContext)();
      await audioContext.resume();
      playbackRate = audioContext.sampleRate;
      console.log("[AUDIO] Playback resumed, rate =", playbackRate);

      // 2) Load & instantiate the AudioWorklet for TTS playback
      await audioContext.audioWorklet.addModule("/static/playback-processor.js");
      playProcessorNode = new AudioWorkletNode(audioContext, "playback-processor");
      playProcessorNode.connect(audioContext.destination);
      console.log("[AUDIO] Worklet loaded & connected");

      // 3) Create & resume capture AudioContext
      captureCtx = new (window.AudioContext || window.webkitAudioContext)();
      await captureCtx.resume();
      micSampleRate = captureCtx.sampleRate;
      console.log("[AUDIO] Capture resumed, rate =", micSampleRate);

      // 4) Kick off WebSocket + mic capture
      await startStreaming();

    } else {
      // ─── USER CLICKED “STOP” ────────────────────────────────────────
      streaming = false;
      startStopBtn.textContent = "Start the Conversation";
      startStopBtn.classList.remove("stop");
      document.getElementById("info").textContent = "";

      // Tear down mic
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

      // Close WS
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.close();
      }

      // Also close both AudioContexts so they don’t linger
      playbackEndTime = 0;
      if (audioContext) {
        audioContext.close();
        audioContext = null;
      }
      if (captureCtx) {
        captureCtx.close();
        captureCtx = null;
      }
    }
  });
});

// ─── Downsample or Upsample Float32Array [srcRate → 16 kHz] ─────────────────────────
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

// ─── Resample Float32Array [TTS Sample Rate → playbackRate] (linear interpolation) ──────────
function resampleTTSToPlayback(bufferTTS) {
  if (playbackRate === TTS_SAMPLE_RATE) {
    return bufferTTS.slice(); // shallow copy if no resampling needed
  }
  const targetLen = Math.round((bufferTTS.length * playbackRate) / TTS_SAMPLE_RATE);
  const result = new Float32Array(targetLen);
  const ratio = TTS_SAMPLE_RATE / playbackRate;
  for (let i = 0; i < targetLen; i++) {
    const idx = i * ratio;
    const i0 = Math.floor(idx);
    const i1 = Math.min(bufferTTS.length - 1, i0 + 1);
    const w = idx - i0;
    result[i] = (1 - w) * bufferTTS[i0] + w * bufferTTS[i1];
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

// ─── When we receive an Int16@TTS SAMPLE RATE, TTS chunk, convert, resample, and send ──────────
function handleBinaryFrame(arrayBuffer) {
  // 1) Read raw 16-bit PCM @ TTS SAMPLE RATE
  console.log("[Client] 🔊 Received TTS chunk, byteLength =", arrayBuffer.byteLength);

  const pcm16 = new Int16Array(arrayBuffer);
  if (!pcm16.length) return;

  // 2) Convert to Float32 @ TTS SAMPLE RATE
  const floatsTTS = new Float32Array(pcm16.length);
  for (let i = 0; i < pcm16.length; i++) {
    floatsTTS[i] = pcm16[i] / 32768;
  }

  // ─── Resample from TTS SAMPLE RATE → playbackRate, if needed ───────────────────
  let toSend;
  if (playbackRate !== TTS_SAMPLE_RATE) {
    toSend = resampleTTSToPlayback(floatsTTS);
  } else {
    toSend = floatsTTS;
  }

  // 3) Post that (possibly resampled) Float32 chunk to the worklet’s port
  playProcessorNode.port.postMessage(toSend);

  // 4) Recompute playbackEndTime:
  //    We assume “toSend.length” samples will be drained in real time at playbackRate.
  const now = audioContext.currentTime;
  playbackEndTime = now + (toSend.length / playbackRate);

  console.log(
    "[DEBUG] TTS chunk arrived (orig", pcm16.length, "samples → resampled",
    toSend.length, "samples). Approx unmute at", playbackEndTime.toFixed(3)
  );
}

// ─── Build WebSocket and attach handlers ───────────────────────────────────────
function setupWebSocket() {
  // Include chosenLanguage as a query parameter
  const protocol = window.location.protocol === "https:" ? "wss" : "ws";
  const url = `${protocol}://${window.location.host}/ws?lang=${encodeURIComponent(selectedLanguage)}`;
  ws = new WebSocket(url);
  ws.binaryType = "arraybuffer";

  ws.onopen = () => {
    console.log("[WS] Connection opened");
    console.log(`[UI] WS opened with ?lang=${selectedLanguage}`);
  };

  ws.onerror = (err) => console.error("[WS] Error:", err);
  ws.onclose = () => console.log("[WS] Connection closed");

  ws.onmessage = (evt) => {
    // if (evt.data instanceof ArrayBuffer) {
    //   // Incoming TTS chunk → handle (convert/resample/send & update end time)
    //   handleBinaryFrame(evt.data);
    // } else if (evt.data instanceof Blob) {
    //   // Some browsers wrap bytes in a Blob
    //   evt.data
    //     .arrayBuffer()
    //     .then((ab) => handleBinaryFrame(ab))
    //     .catch((e) => console.error("[WS] Blob→ArrayBuffer error:", e));
    // } else {
    //   // JSON control message: transcript or token
    //   try {
    //     const msg = JSON.parse(evt.data);
    //     if (msg.type === "transcript") {
    //       const label = msg.final ? "FINAL" : "INTERIM";
    //       document.getElementById("transcripts").textContent +=
    //         `\nTRANSCRIPT [${label}]: ${msg.text}\n`;
    //     } else if (msg.type === "token") {
    //       document.getElementById("transcripts").textContent += msg.token;
    //     }
    //   } catch {
    //     console.warn("[WS] Non-JSON message:", evt.data);
    //   }
    // }

    // 1) Binary frames = TTS audio
    if (evt.data instanceof ArrayBuffer || evt.data instanceof Blob) {
      const handle = (ab) => handleBinaryFrame(ab);
      if (evt.data instanceof Blob) {
        evt.data.arrayBuffer().then(handle).catch(console.error);
      } else {
        handle(evt.data);
      }
      return;
    }

    // 2) JSON control messages
    let msg;
    try {
      msg = JSON.parse(evt.data);
    } catch {
      console.warn("[WS] Non-JSON message:", evt.data);
      return;
    }

    if (msg.type === "stop_speech") {
      // Flush worklet buffer immediately
      playProcessorNode.port.postMessage({ command: "flush" });
      // Allow mic capture right away
      playbackEndTime = 0;
      console.log("[Client] Received stop_speech → flushing audio");
      return;
    }

    if (msg.type === "transcript") {
      const label = msg.final ? "FINAL" : "INTERIM";
      document.getElementById("transcripts").textContent +=
        `\nTRANSCRIPT [${label}]: ${msg.text}\n`;
    }
    else if (msg.type === "token") {
      document.getElementById("transcripts").textContent += msg.token;
    }
  };

  return ws;
}

// ─── Start capturing mic → WS, skip when currentTime < playbackEndTime ─────────
function startMicStreaming(ws) {
  const bufferSize = 4096;
  scriptNodeSender = audioContext.createScriptProcessor(bufferSize, 1, 1);
  const micSource = audioContext.createMediaStreamSource(micStream);
  micSource.connect(scriptNodeSender);

  scriptNodeSender.onaudioprocess = (event) => {
    // If the currentTime is still before playbackEndTime + 50 ms, drop mic
    if (audioContext.currentTime < playbackEndTime + 0.05) {
      return;
    }

    // Otherwise, capture mic floats as before
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

    // Send 320‐sample Int16 frames
    let i = 0;
    while (i + 320 <= down16k.length) {
      const slice = down16k.subarray(i, i + 320);
      const int16 = floatToInt16(slice);
      if (ws.readyState === WebSocket.OPEN) {
        ws.send(int16.buffer);
      }
      i += 320;
    }

    // Keep leftover for next round
    const leftoverIn = Math.round((down16k.length - i) * (micSampleRate / TARGET_SAMPLE_RATE));
    const leftover = merged.subarray(merged.length - leftoverIn);
    float32Queue = leftoverIn > 0 ? [leftover] : [];
  };

  // Connect to destination so the node stays alive (no output)
  scriptNodeSender.connect(audioContext.destination);
  console.log("[MIC] Mic streaming started");
}

// ─── Start everything when user clicks “Start” ─────────────────────────────────
async function startStreaming() {
  // 1) Resume AudioContext if needed
  if (audioContext.state === "suspended") {
    await audioContext.resume();
    console.log("[DEBUG] audioContext resumed; state =", audioContext.state);
  }

  // 2) Request microphone
  try {
    micStream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true },
    });
    micSampleRate = captureCtx.sampleRate;
    console.log("[DEBUG] Mic sampleRate =", micSampleRate);
  } catch (err) {
    console.error("[UI] getUserMedia error:", err);
    return;
  }

  // 3) Open WebSocket (includes ?lang=<selectedLanguage>)
  const socket = setupWebSocket();

  // 4) Once WS is open, start mic streaming
  socket.addEventListener("open", () => {
    console.log("[WS] Ready → starting mic streaming");
    startMicStreaming(socket);
  });
} 