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
          setPermissionDenied(true);
          setError(
            "Microphone access was blocked. If this app is embedded in a page, open it directly and allow the microphone."
          );
          break;
        case "no-speech":
          setError("No speech detected — try again.");
          break;
        case "audio-capture":
          setError("No microphone was found.");
          break;
        case "network":
          setError("Speech recognition needs a network connection.");
          break;
        case "aborted":
          // Fired by our own stop(); not an error worth showing.
          break;
        default:
          setError(`Speech recognition error: ${event.error}`);
      }
    };

    recognition.onend = () => {
      setInterim("");
      if (wantListeningRef.current) {
        // Restart seamlessly so a pause doesn't end the session.
        try {
          recognition.start();
          return;
        } catch {
          // start() throws if already started; fall through to stopped state.
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
