"use client";

/**
 * Voice input via the browser-native Web Speech API.
 *
 * No backend, no API key, no audio leaves the device through our servers — the
 * browser performs recognition and hands back text. Supported in Chrome, Edge
 * and Safari; Firefox does not implement SpeechRecognition, so `supported` is
 * false there and callers should hide the microphone.
 *
 * Limitation to be aware of: microphone access is blocked when the page runs
 * inside a cross-origin iframe without an `allow="microphone"` permission
 * policy (for example the Hugging Face Space page that embeds this app).
 * Opening the Space URL directly works. We surface that as `permissionDenied`.
 */

import { useCallback, useEffect, useRef, useState } from "react";

// The Web Speech API is not in every TypeScript DOM lib yet, so declare the
// minimal surface we use instead of relying on `SpeechRecognition` existing.
type SpeechRecognitionAlternative = { transcript: string; confidence: number };
type SpeechRecognitionResult = {
  readonly length: number;
  readonly isFinal: boolean;
  item(index: number): SpeechRecognitionAlternative;
  [index: number]: SpeechRecognitionAlternative;
};
type SpeechRecognitionResultList = {
  readonly length: number;
  item(index: number): SpeechRecognitionResult;
  [index: number]: SpeechRecognitionResult;
};
type SpeechRecognitionEventLike = {
  readonly resultIndex: number;
  readonly results: SpeechRecognitionResultList;
};
type SpeechRecognitionErrorEventLike = {
  readonly error: string;
  readonly message?: string;
};

type SpeechRecognitionLike = {
  lang: string;
  continuous: boolean;
  interimResults: boolean;
  maxAlternatives: number;
  start(): void;
  stop(): void;
  abort(): void;
  onresult: ((event: SpeechRecognitionEventLike) => void) | null;
  onerror: ((event: SpeechRecognitionErrorEventLike) => void) | null;
  onend: (() => void) | null;
  onstart: (() => void) | null;
};

type SpeechRecognitionCtor = new () => SpeechRecognitionLike;

function getRecognitionCtor(): SpeechRecognitionCtor | null {
  if (typeof window === "undefined") return null;
  const w = window as unknown as {
    SpeechRecognition?: SpeechRecognitionCtor;
    webkitSpeechRecognition?: SpeechRecognitionCtor;
  };
  return w.SpeechRecognition ?? w.webkitSpeechRecognition ?? null;
}

export type UseSpeechRecognitionOptions = {
  /** Called with each newly-finalised chunk of transcript. */
  onTranscript: (text: string) => void;
  /** BCP-47 language tag. Defaults to the browser's language. */
  lang?: string;
};

export type UseSpeechRecognitionResult = {
  /** Whether the browser implements the Web Speech API at all. */
  supported: boolean;
  /** True while actively capturing speech. */
  listening: boolean;
  /** Live (not yet final) words, for display while speaking. */
  interim: string;
  /** Human-readable error, cleared on the next start(). */
  error: string | null;
  /** True when the browser denied microphone access. */
  permissionDenied: boolean;
  start: () => void;
  stop: () => void;
  toggle: () => void;
};

export function useSpeechRecognition({
  onTranscript,
  lang,
}: UseSpeechRecognitionOptions): UseSpeechRecognitionResult {
  const [supported, setSupported] = useState(false);
  const [listening, setListening] = useState(false);
  const [interim, setInterim] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [permissionDenied, setPermissionDenied] = useState(false);

  const recognitionRef = useRef<SpeechRecognitionLike | null>(null);
  // Kept in a ref so the recognition callbacks (created once) always see the
  // latest handler without re-creating the recogniser.
  const onTranscriptRef = useRef(onTranscript);
  // True while the user wants to be listening, so `onend` can auto-restart
  // (Chrome ends the session after a pause even with continuous = true).
  const wantListeningRef = useRef(false);
  // Set when a failure is not recoverable by restarting. Without this, `onend`
  // restarts after every error and the recogniser spins in a silent loop: the
  // UI keeps saying "Listening…" while no audio is ever transcribed and the
  // real error never surfaces. Cleared on each start().
  const fatalErrorRef = useRef(false);
  // Guards against a tight restart loop when the recogniser ends immediately.
  const restartCountRef = useRef(0);
  const lastRestartRef = useRef(0);

  useEffect(() => {
    onTranscriptRef.current = onTranscript;
  }, [onTranscript]);

  useEffect(() => {
    const Ctor = getRecognitionCtor();
    setSupported(Boolean(Ctor));
    if (!Ctor) return;

    const recognition = new Ctor();
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.maxAlternatives = 1;
    recognition.lang =
      lang ??
      (typeof navigator !== "undefined" && navigator.language
        ? navigator.language
        : "en-US");

    recognition.onstart = () => {
      setListening(true);
      setError(null);
    };

    recognition.onresult = (event) => {
      let finalText = "";
      let interimText = "";
      for (let i = event.resultIndex; i < event.results.length; i += 1) {
        const result = event.results[i];
        const text = result[0]?.transcript ?? "";
        if (result.isFinal) finalText += text;
        else interimText += text;
      }
      if (finalText) {
        onTranscriptRef.current(finalText.trim());
      }
      setInterim(interimText);
    };

    recognition.onerror = (event) => {
      switch (event.error) {
        case "not-allowed":
        case "service-not-allowed":
          fatalErrorRef.current = true;
          setPermissionDenied(true);
          setError(
            "Microphone access was blocked. Allow the microphone for this site, and if the app is embedded in another page, open it directly."
          );
          break;
        case "no-speech":
          // Recoverable: the user simply did not speak. Let onend restart.
          setError("No speech detected — try again.");
          break;
        case "audio-capture":
          fatalErrorRef.current = true;
          setError("No microphone was found. Check that one is connected and enabled.");
          break;
        case "network":
          // Chrome's recogniser talks to Google's speech service. If that is
          // unreachable (offline, blocked, or unreachable region) recognition
          // cannot work — restarting only loops silently, so stop retrying.
          fatalErrorRef.current = true;
          setError(
            "Speech recognition could not reach the browser's speech service. Check your connection, or type your question instead."
          );
          break;
        case "aborted":
          // Fired by our own stop(); not an error worth showing.
          break;
        case "language-not-supported":
          fatalErrorRef.current = true;
          setError("Speech recognition does not support this language.");
          break;
        default:
          fatalErrorRef.current = true;
          setError(`Speech recognition error: ${event.error}`);
      }
    };

    recognition.onend = () => {
      setInterim("");
      // Never auto-restart after a fatal error, and never restart in a tight
      // loop: if sessions keep ending within a second of starting, back off.
      if (wantListeningRef.current && !fatalErrorRef.current) {
        const now = Date.now();
        if (now - lastRestartRef.current < 1000) {
          restartCountRef.current += 1;
        } else {
          restartCountRef.current = 0;
        }
        lastRestartRef.current = now;

        if (restartCountRef.current < 3) {
          try {
            recognition.start();
            return;
          } catch {
            // start() throws if already started; fall through to stopped state.
          }
        } else {
          // Repeated immediate restarts mean the recogniser cannot run here.
          wantListeningRef.current = false;
          setError(
            "Speech recognition keeps stopping immediately and cannot be used here. Please type your question instead."
          );
        }
      }
      setListening(false);
    };

    recognitionRef.current = recognition;

    return () => {
      wantListeningRef.current = false;
      recognition.onresult = null;
      recognition.onerror = null;
      recognition.onend = null;
      recognition.onstart = null;
      try {
        recognition.abort();
      } catch {
        // Ignore: aborting a non-started recogniser throws in some browsers.
      }
      recognitionRef.current = null;
    };
  }, [lang]);

  const start = useCallback(() => {
    const recognition = recognitionRef.current;
    if (!recognition) return;
    setError(null);
    setPermissionDenied(false);
    // A fresh user action clears any previous fatal state so the user can
    // retry after fixing the cause (e.g. granting the microphone).
    fatalErrorRef.current = false;
    restartCountRef.current = 0;
    lastRestartRef.current = 0;
    wantListeningRef.current = true;
    try {
      recognition.start();
    } catch {
      // start() throws if already running; treat as already listening.
      setListening(true);
    }
  }, []);

  const stop = useCallback(() => {
    const recognition = recognitionRef.current;
    wantListeningRef.current = false;
    setInterim("");
    if (!recognition) return;
    try {
      recognition.stop();
    } catch {
      // Ignore.
    }
    setListening(false);
  }, []);

  const toggle = useCallback(() => {
    if (wantListeningRef.current) stop();
    else start();
  }, [start, stop]);

  return { supported, listening, interim, error, permissionDenied, start, stop, toggle };
}
