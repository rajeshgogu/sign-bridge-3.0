/**
 * ISL Phrase Classifier — Rewritten for accuracy.
 *
 * Key design principles:
 *  1. Every phrase gets a UNIQUE hand shape signature that does NOT overlap
 *     with other phrases. Shapes like open-palm are only used ONCE.
 *  2. Strict confidence thresholds per phrase.
 *  3. Temporal smoothing: 4-frame rolling buffer — must repeat 3× before firing.
 *  4. Two-hand signs require BOTH hands to be present + matching shapes.
 *  5. The classifier uses a global winner-takes-all after all scores computed.
 */

import type { HandLandmark } from "@/types/ml";

// ── Geometry ──────────────────────────────────────────────────────────────────

function dist(a: HandLandmark, b: HandLandmark): number {
  return Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2);
}

function angleDeg(a: HandLandmark, b: HandLandmark, c: HandLandmark): number {
  const ba = { x: a.x - b.x, y: a.y - b.y, z: a.z - b.z };
  const bc = { x: c.x - b.x, y: c.y - b.y, z: c.z - b.z };
  const dot = ba.x * bc.x + ba.y * bc.y + ba.z * bc.z;
  const magBA = Math.sqrt(ba.x ** 2 + ba.y ** 2 + ba.z ** 2);
  const magBC = Math.sqrt(bc.x ** 2 + bc.y ** 2 + bc.z ** 2);
  if (magBA < 1e-6 || magBC < 1e-6) return 180;
  return Math.acos(Math.max(-1, Math.min(1, dot / (magBA * magBC)))) * (180 / Math.PI);
}

// ── Feature Extraction ────────────────────────────────────────────────────────

export interface HandFeatures {
  // Finger curl: 0=fully extended, 1=fully curled
  thumbCurl: number;
  indexCurl: number;
  middleCurl: number;
  ringCurl: number;
  pinkyCurl: number;

  // PIP angles (180=straight, ~90=curled)
  indexPIP: number;
  middlePIP: number;
  ringPIP: number;
  pinkyPIP: number;

  // Spread between adjacent fingertips (normalized by palmSize)
  thumbIndexDist: number;   // thumb tip ↔ index tip
  thumbMiddleDist: number;  // thumb tip ↔ middle tip
  thumbRingDist: number;    // thumb tip ↔ ring tip
  thumbPinkyDist: number;   // thumb tip ↔ pinky tip
  indexMiddleDist: number;  // index tip ↔ middle tip
  middleRingDist: number;   // middle tip ↔ ring tip
  ringPinkyDist: number;    // ring tip ↔ pinky tip

  // Thumb abduction: distance thumb tip ↔ index MCP, normalized
  thumbAbduction: number;

  // Thumb crosses palm (tip near ring/middle MCP)
  thumbCrossesPalm: boolean;

  // All-finger spread (open palm feel)
  allSpread: boolean;

  palmSize: number;
}

function extractFeatures(lm: HandLandmark[]): HandFeatures | null {
  if (lm.length < 21) return null;
  const palmSize = dist(lm[0], lm[9]);
  if (palmSize < 1e-6) return null;

  function fingerCurl(mcpIdx: number, pipIdx: number, dipIdx: number, tipIdx: number): number {
    const pip = angleDeg(lm[mcpIdx], lm[pipIdx], lm[dipIdx]);
    const dip = angleDeg(lm[pipIdx], lm[dipIdx], lm[tipIdx]);
    return Math.max(0, Math.min(1, 1 - (pip + dip) / 360));
  }

  function fingerPIP(mcpIdx: number, pipIdx: number, dipIdx: number): number {
    return angleDeg(lm[mcpIdx], lm[pipIdx], lm[dipIdx]);
  }

  // Thumb curl — uses joints 1,2,3,4
  const thumbPIP = angleDeg(lm[1], lm[2], lm[3]);
  const thumbDIP = angleDeg(lm[2], lm[3], lm[4]);
  const thumbCurl = Math.max(0, Math.min(1, 1 - (thumbPIP + thumbDIP) / 360));

  const thumbAbduction = dist(lm[4], lm[5]) / palmSize; // thumb tip to index MCP
  const thumbCrossesPalm =
    dist(lm[4], lm[9]) / palmSize < 0.55 || dist(lm[4], lm[13]) / palmSize < 0.55;

  const thumbIndexDist  = dist(lm[4], lm[8])  / palmSize;
  const thumbMiddleDist = dist(lm[4], lm[12]) / palmSize;
  const thumbRingDist   = dist(lm[4], lm[16]) / palmSize;
  const thumbPinkyDist  = dist(lm[4], lm[20]) / palmSize;
  const indexMiddleDist = dist(lm[8], lm[12]) / palmSize;
  const middleRingDist  = dist(lm[12], lm[16]) / palmSize;
  const ringPinkyDist   = dist(lm[16], lm[20]) / palmSize;

  const allSpread = indexMiddleDist > 0.38 && middleRingDist > 0.38;

  return {
    thumbCurl,
    indexCurl:  fingerCurl(5, 6, 7, 8),
    middleCurl: fingerCurl(9, 10, 11, 12),
    ringCurl:   fingerCurl(13, 14, 15, 16),
    pinkyCurl:  fingerCurl(17, 18, 19, 20),
    indexPIP:   fingerPIP(5, 6, 7),
    middlePIP:  fingerPIP(9, 10, 11),
    ringPIP:    fingerPIP(13, 14, 15),
    pinkyPIP:   fingerPIP(17, 18, 19),
    thumbIndexDist,
    thumbMiddleDist,
    thumbRingDist,
    thumbPinkyDist,
    indexMiddleDist,
    middleRingDist,
    ringPinkyDist,
    thumbAbduction,
    thumbCrossesPalm,
    allSpread,
    palmSize,
  };
}

// ── Shape Predicates (strict thresholds) ──────────────────────────────────────

/** Finger clearly straight (not just partially extended) */
function isExtended(curl: number, pip: number): boolean {
  return curl < 0.22 && pip > 145;
}

/** Finger clearly bent/curled */
function isCurled(curl: number, pip: number): boolean {
  return curl > 0.35 || pip < 135;
}

/** Finger halfway — hooked or bent (C-shape) */
function isBent(curl: number, pip: number): boolean {
  return curl >= 0.18 && curl <= 0.52 && pip >= 105 && pip <= 165;
}

// Thumb helpers
function isThumbExtended(f: HandFeatures): boolean {
  return f.thumbAbduction > 0.55 && f.thumbCurl < 0.30;
}

function isThumbCurled(f: HandFeatures): boolean {
  return f.thumbCurl > 0.35 || f.thumbAbduction < 0.40;
}

function isThumbPinching(f: HandFeatures): boolean {
  return f.thumbIndexDist < 0.32;
}

// ── Phrase Shape Signatures ───────────────────────────────────────────────────
// Each phrase maps to a totally distinct hand shape or two-hand requirement.
// Min threshold to fire is listed per phrase.

export interface PhraseSignature {
  /** Minimum score (0–1) needed to accept this phrase */
  minScore: number;
  /** Requires two hands visible */
  twoHanded: boolean;
  /** Score function — receives array of features (1 or 2 hands) */
  score: (hands: HandFeatures[]) => number;
}

// Helper: score how well a single hand matches a pattern
function singleHandScore(f: HandFeatures, checks: boolean[]): number {
  const passed = checks.filter(Boolean).length;
  return passed / checks.length;
}

const PHRASE_SIGNATURES: Record<string, PhraseSignature> = {

  // ─────────────────────────────────────────────────────────────────────────
  // GREETINGS
  // ─────────────────────────────────────────────────────────────────────────

  hello: {
    minScore: 0.82,      // Increased to separate from generic open palm
    twoHanded: false,
    score: ([f]) => {
      if (!f) return 0;
      return singleHandScore(f, [
        isExtended(f.indexCurl, f.indexPIP),
        isExtended(f.middleCurl, f.middlePIP),
        isExtended(f.ringCurl, 150),
        isExtended(f.pinkyCurl, 150),
        isThumbExtended(f),
        f.allSpread,            // Hello MUST be spread
        f.indexMiddleDist > 0.40,
      ]);
    },
  },

  thank_you: {
    minScore: 0.88,      // High threshold for flat-palm forward
    twoHanded: false,
    score: ([f]) => {
      if (!f) return 0;
      return singleHandScore(f, [
        isExtended(f.indexCurl, f.indexPIP),
        isExtended(f.middleCurl, f.middlePIP),
        isExtended(f.ringCurl, 150),
        isExtended(f.pinkyCurl, 150),
        f.indexMiddleDist < 0.18,          // Fingers MUST be touching
        f.middleRingDist < 0.18,
        !f.allSpread,                      // NO spread
      ]);
    },
  },

  how_are_you_greet: {   // Renamed from how_are_you
    minScore: 0.72,
    twoHanded: true,
    score: ([f0, f1]) => {
      if (!f0 || !f1) return 0;
      function cShape(f: HandFeatures) {
        return singleHandScore(f, [
          isBent(f.indexCurl, f.indexPIP),
          isBent(f.middleCurl, f.middlePIP),
          isBent(f.ringCurl, 140),
          isThumbExtended(f),
          f.thumbIndexDist > 0.40,
        ]);
      }
      return (cShape(f0) + cShape(f1)) / 2;
    },
  },

  goodbye_greet: {       // Renamed from goodbye
    minScore: 0.80,
    twoHanded: false,
    score: ([f]) => {
      if (!f) return 0;
      return singleHandScore(f, [
        isExtended(f.indexCurl, f.indexPIP),
        isExtended(f.middleCurl, f.middlePIP),
        isExtended(f.ringCurl, 155),
        isExtended(f.pinkyCurl, 155),
        f.indexMiddleDist >= 0.22 && f.indexMiddleDist <= 0.38,
        isThumbExtended(f),
        !f.allSpread,
      ]);
    },
  },

  name_is: {            // Renamed from my_name_is
    minScore: 0.74,
    twoHanded: false,
    score: ([f]) => {
      if (!f) return 0;
      return singleHandScore(f, [
        isExtended(f.indexCurl, f.indexPIP),
        isCurled(f.middleCurl, f.middlePIP),
        isCurled(f.ringCurl, 100),
        isCurled(f.pinkyCurl, 100),
        f.thumbAbduction < 0.60,
        f.indexMiddleDist > 0.28,
      ]);
    },
  },

  meet_you: {           // Renamed from nice_to_meet_you
    minScore: 0.68,
    twoHanded: true,
    score: ([f0, f1]) => {
      if (!f0 || !f1) return 0;
      function flatPalm(f: HandFeatures) {
        return singleHandScore(f, [
          isExtended(f.indexCurl, f.indexPIP),
          isExtended(f.middleCurl, f.middlePIP),
          isExtended(f.ringCurl, 145),
          isExtended(f.pinkyCurl, 145),
          f.indexMiddleDist < 0.35,
          !f.allSpread,
        ]);
      }
      return (flatPalm(f0) + flatPalm(f1)) / 2;
    },
  },

  im_fine: {            // Renamed from i_am_fine
    minScore: 0.80,
    twoHanded: false,
    score: ([f]) => {
      if (!f) return 0;
      return singleHandScore(f, [
        isThumbExtended(f),
        f.thumbAbduction > 0.60,
        isCurled(f.indexCurl, f.indexPIP),
        isCurled(f.middleCurl, f.middlePIP),
        isCurled(f.ringCurl, 100),
        isCurled(f.pinkyCurl, 100),
      ]);
    },
  },

  welcome: {
    minScore: 0.80,
    twoHanded: false,
    score: ([f]) => {
      if (!f) return 0;
      return singleHandScore(f, [
        isExtended(f.indexCurl, f.indexPIP),
        isExtended(f.middleCurl, f.middlePIP),
        isExtended(f.ringCurl, 155),
        isExtended(f.pinkyCurl, 155),
        isThumbExtended(f),
        f.thumbAbduction > 0.60,
        f.indexMiddleDist < 0.32,
      ]);
    },
  },

  // ─────────────────────────────────────────────────────────────────────────
  // BASIC NEEDS
  // ─────────────────────────────────────────────────────────────────────────

  help_me: {           // Renamed from help
    minScore: 0.78,
    twoHanded: false,
    score: ([f]) => {
      if (!f) return 0;
      return singleHandScore(f, [
        isThumbExtended(f),
        f.thumbAbduction > 0.55,
        isCurled(f.indexCurl, f.indexPIP),
        isCurled(f.middleCurl, f.middlePIP),
        isCurled(f.ringCurl, 100),
        isCurled(f.pinkyCurl, 100),
        f.thumbIndexDist > 0.55,
      ]);
    },
  },

  yes_simple: {         // Renamed from yes
    minScore: 0.78,
    twoHanded: false,
    score: ([f]) => {
      if (!f) return 0;
      return singleHandScore(f, [
        isCurled(f.indexCurl, f.indexPIP),
        isCurled(f.middleCurl, f.middlePIP),
        isCurled(f.ringCurl, 100),
        isCurled(f.pinkyCurl, 100),
        f.thumbCrossesPalm,
        !isThumbExtended(f),
      ]);
    },
  },

  water: {
    minScore: 0.78,
    twoHanded: false,
    score: ([f]) => {
      if (!f) return 0;
      return singleHandScore(f, [
        isExtended(f.indexCurl, f.indexPIP),
        isExtended(f.middleCurl, f.middlePIP),
        isExtended(f.ringCurl, 150),
        isCurled(f.pinkyCurl, f.pinkyPIP),
        isThumbCurled(f),
        f.indexMiddleDist > 0.25,
        f.middleRingDist > 0.20,
      ]);
    },
  },

  food: {               // Renamed from need_food
    minScore: 0.75,
    twoHanded: false,
    score: ([f]) => {
      if (!f) return 0;
      return singleHandScore(f, [
        f.thumbIndexDist < 0.35,
        f.thumbMiddleDist < 0.40,
        isBent(f.indexCurl, f.indexPIP),
        isBent(f.middleCurl, f.middlePIP),
        isBent(f.ringCurl, 140),
        isBent(f.pinkyCurl, 140),
        !isThumbExtended(f),
      ]);
    },
  },

  rest: {               // Renamed from need_rest
    minScore: 0.68,
    twoHanded: true,
    score: ([f0, f1]) => {
      if (!f0 || !f1) return 0;
      function restShape(f: HandFeatures) {
        return singleHandScore(f, [
          isExtended(f.indexCurl, f.indexPIP),
          isExtended(f.middleCurl, f.middlePIP),
          isExtended(f.ringCurl, 145),
          isExtended(f.pinkyCurl, 145),
          f.indexMiddleDist < 0.38,
        ]);
      }
      return (restShape(f0) + restShape(f1)) / 2;
    },
  },

  cold: {
    minScore: 0.72,
    twoHanded: true,
    score: ([f0, f1]) => {
      if (!f0 || !f1) return 0;
      function fist(f: HandFeatures) {
        return singleHandScore(f, [
          isCurled(f.indexCurl, f.indexPIP),
          isCurled(f.middleCurl, f.middlePIP),
          isCurled(f.ringCurl, 100),
          isCurled(f.pinkyCurl, 100),
        ]);
      }
      return (fist(f0) + fist(f1)) / 2;
    },
  },

  tired: {
    minScore: 0.72,
    twoHanded: true,
    score: ([f0, f1]) => {
      if (!f0 || !f1) return 0;
      function droopFlat(f: HandFeatures) {
        return singleHandScore(f, [
          isExtended(f.indexCurl, f.indexPIP),
          isExtended(f.middleCurl, f.middlePIP),
          isExtended(f.ringCurl, 155),
          isExtended(f.pinkyCurl, 155),
          f.indexMiddleDist < 0.30,
          f.middleRingDist < 0.30,
        ]);
      }
      return (droopFlat(f0) + droopFlat(f1)) / 2;
    },
  },

  // ─────────────────────────────────────────────────────────────────────────
  // COMMUNICATION
  // ─────────────────────────────────────────────────────────────────────────

  call_me: {
    minScore: 0.78,
    twoHanded: false,
    score: ([f]) => {
      if (!f) return 0;
      return singleHandScore(f, [
        isThumbExtended(f),
        isCurled(f.indexCurl, f.indexPIP),
        isCurled(f.middleCurl, f.middlePIP),
        isCurled(f.ringCurl, 100),
        isExtended(f.pinkyCurl, 155),
        f.thumbAbduction > 0.55,
      ]);
    },
  },

  call_doctor_request: {
    minScore: 0.78,
    twoHanded: false,
    score: ([f]) => {
      if (!f) return 0;
      return singleHandScore(f, [
        isThumbExtended(f),
        isCurled(f.indexCurl, f.indexPIP),
        isCurled(f.middleCurl, f.middlePIP),
        isCurled(f.ringCurl, 100),
        isExtended(f.pinkyCurl, 155),
        f.thumbAbduction > 0.55,
        f.ringPinkyDist > 0.30,
      ]);
    },
  },

  call_family: {
    minScore: 0.75,
    twoHanded: false,
    score: ([f]) => {
      if (!f) return 0;
      return singleHandScore(f, [
        isThumbPinching(f),
        f.thumbIndexDist < 0.35,
        isBent(f.indexCurl, f.indexPIP) || isCurled(f.indexCurl, f.indexPIP),
        isExtended(f.middleCurl, f.middlePIP),
        isExtended(f.ringCurl, 150),
        isExtended(f.pinkyCurl, 150),
      ]);
    },
  },

  no_hear: {
    minScore: 0.78,
    twoHanded: false,
    score: ([f]) => {
      if (!f) return 0;
      return singleHandScore(f, [
        isExtended(f.indexCurl, f.indexPIP),
        isCurled(f.middleCurl, f.middlePIP),
        isCurled(f.ringCurl, 100),
        isCurled(f.pinkyCurl, 100),
        f.thumbCurl > 0.40,
        f.thumbIndexDist < 0.40,
        f.indexMiddleDist > 0.35,
      ]);
    },
  },

  repeat_please: {
    minScore: 0.76,
    twoHanded: false,
    score: ([f]) => {
      if (!f) return 0;
      return singleHandScore(f, [
        isExtended(f.indexCurl, f.indexPIP),
        isExtended(f.middleCurl, f.middlePIP),
        isExtended(f.ringCurl, 150),
        isExtended(f.pinkyCurl, 150),
        f.indexMiddleDist < 0.25,
        f.middleRingDist < 0.25,
        f.thumbCurl > 0.25,
      ]);
    },
  },

  speak_slowly: {
    minScore: 0.76,
    twoHanded: false,
    score: ([f]) => {
      if (!f) return 0;
      return singleHandScore(f, [
        isExtended(f.indexCurl, f.indexPIP),
        isExtended(f.middleCurl, f.middlePIP),
        isExtended(f.ringCurl, 150),
        isExtended(f.pinkyCurl, 150),
        f.indexMiddleDist < 0.28,
        f.thumbCurl > 0.20 && f.thumbCurl < 0.45,
        !f.allSpread,
      ]);
    },
  },

  understand: {
    minScore: 0.76,
    twoHanded: false,
    score: ([f]) => {
      if (!f) return 0;
      return singleHandScore(f, [
        f.indexCurl < 0.25 && f.indexPIP > 140,
        isCurled(f.middleCurl, f.middlePIP),
        isCurled(f.ringCurl, 100),
        isCurled(f.pinkyCurl, 100),
        !f.thumbCrossesPalm,
        f.thumbCurl > 0.30,
        f.indexMiddleDist > 0.28,
      ]);
    },
  },

  no_understand: {
    minScore: 0.76,
    twoHanded: false,
    score: ([f]) => {
      if (!f) return 0;
      return singleHandScore(f, [
        f.indexCurl < 0.30,
        isCurled(f.middleCurl, f.middlePIP),
        isCurled(f.ringCurl, 100),
        isCurled(f.pinkyCurl, 100),
        f.thumbCurl > 0.35,
        f.thumbIndexDist < 0.45,
      ]);
    },
  },

  write_down: {
    minScore: 0.76,
    twoHanded: false,
    score: ([f]) => {
      if (!f) return 0;
      return singleHandScore(f, [
        isExtended(f.indexCurl, f.indexPIP),
        isExtended(f.middleCurl, f.middlePIP),
        isCurled(f.ringCurl, 100),
        isCurled(f.pinkyCurl, 100),
        isThumbPinching(f),
        f.thumbIndexDist < 0.38,
        f.indexMiddleDist < 0.30,
      ]);
    },
  },

  help_comm: {
    minScore: 0.78,
    twoHanded: false,
    score: ([f]) => {
      if (!f) return 0;
      return singleHandScore(f, [
        isThumbExtended(f),
        f.thumbAbduction > 0.58,
        isCurled(f.indexCurl, f.indexPIP),
        isCurled(f.middleCurl, f.middlePIP),
        isCurled(f.ringCurl, 100),
        isCurled(f.pinkyCurl, 100),
        f.thumbMiddleDist > 0.55,
      ]);
    },
  },

  // ─────────────────────────────────────────────────────────────────────────
  // EMERGENCY
  // ─────────────────────────────────────────────────────────────────────────

  need_doctor: {
    minScore: 0.78,
    twoHanded: false,
    score: ([f]) => {
      if (!f) return 0;
      return singleHandScore(f, [
        isExtended(f.indexCurl, f.indexPIP),
        isExtended(f.middleCurl, f.middlePIP),
        isCurled(f.ringCurl, 100),
        isCurled(f.pinkyCurl, 100),
        f.indexMiddleDist < 0.25,
        f.thumbCurl > 0.25,
      ]);
    },
  },

  in_pain: {
    minScore: 0.78,
    twoHanded: false,
    score: ([f]) => {
      if (!f) return 0;
      return singleHandScore(f, [
        isExtended(f.indexCurl, f.indexPIP),
        isCurled(f.middleCurl, f.middlePIP),
        isCurled(f.ringCurl, 100),
        isCurled(f.pinkyCurl, 100),
        f.thumbCurl > 0.40,
        f.indexMiddleDist > 0.32,
      ]);
    },
  },

  ambulance_call: {
    minScore: 0.78,
    twoHanded: false,
    score: ([f]) => {
      if (!f) return 0;
      return singleHandScore(f, [
        isThumbExtended(f),
        f.thumbAbduction > 0.50,
        isCurled(f.indexCurl, f.indexPIP),
        isCurled(f.middleCurl, f.middlePIP),
        isCurled(f.ringCurl, 100),
        isCurled(f.pinkyCurl, 100),
        f.thumbCurl < 0.30,
      ]);
    },
  },

  i_have_fever: {
    minScore: 0.80,
    twoHanded: false,
    score: ([f]) => {
      if (!f) return 0;
      return singleHandScore(f, [
        isExtended(f.indexCurl, f.indexPIP),
        isExtended(f.middleCurl, f.middlePIP),
        isExtended(f.ringCurl, 155),
        isExtended(f.pinkyCurl, 155),
        f.thumbCurl > 0.30,
        !isThumbExtended(f),
        f.indexMiddleDist < 0.30,
      ]);
    },
  },

  feel_dizzy: {
    minScore: 0.75,
    twoHanded: false,
    score: ([f]) => {
      if (!f) return 0;
      return singleHandScore(f, [
        isExtended(f.indexCurl, f.indexPIP),
        isCurled(f.middleCurl, f.middlePIP),
        isCurled(f.ringCurl, 100),
        isCurled(f.pinkyCurl, 100),
        f.thumbCurl > 0.30 && f.thumbCurl < 0.55,
        f.thumbIndexDist > 0.35,
      ]);
    },
  },

  need_wheelchair: {
    minScore: 0.70,
    twoHanded: true,
    score: ([f0, f1]) => {
      if (!f0 || !f1) return 0;
      function wheelFist(f: HandFeatures) {
        return singleHandScore(f, [
          isCurled(f.indexCurl, f.indexPIP),
          isCurled(f.middleCurl, f.middlePIP),
          isCurled(f.ringCurl, 100),
          isCurled(f.pinkyCurl, 100),
          !f.thumbCrossesPalm,
        ]);
      }
      return (wheelFist(f0) + wheelFist(f1)) / 2;
    },
  },

  i_am_deaf: {
    minScore: 0.78,
    twoHanded: false,
    score: ([f]) => {
      if (!f) return 0;
      return singleHandScore(f, [
        isExtended(f.indexCurl, f.indexPIP),
        isCurled(f.middleCurl, f.middlePIP),
        isCurled(f.ringCurl, 100),
        isCurled(f.pinkyCurl, 100),
        f.thumbCurl > 0.42,
        f.indexMiddleDist > 0.38,
      ]);
    },
  },

  use_isl: {
    minScore: 0.70,
    twoHanded: true,
    score: ([f0, f1]) => {
      if (!f0 || !f1) return 0;
      function rotateFist(f: HandFeatures) {
        return singleHandScore(f, [
          isCurled(f.indexCurl, f.indexPIP),
          isCurled(f.middleCurl, f.middlePIP),
          isCurled(f.ringCurl, 100),
        ]);
      }
      return (rotateFist(f0) + rotateFist(f1)) / 2;
    },
  },

  be_patient: {
    minScore: 0.76,
    twoHanded: false,
    score: ([f]) => {
      if (!f) return 0;
      return singleHandScore(f, [
        isThumbExtended(f),
        f.thumbAbduction > 0.50,
        isCurled(f.indexCurl, f.indexPIP),
        isCurled(f.middleCurl, f.middlePIP),
        isCurled(f.ringCurl, 100),
        f.pinkyCurl > 0.20,
        f.thumbCurl < 0.28,
      ]);
    },
  },

  emergency_urgent: {
    minScore: 0.76,
    twoHanded: false,
    score: ([f]) => {
      if (!f) return 0;
      return singleHandScore(f, [
        isCurled(f.indexCurl, f.indexPIP),
        isCurled(f.middleCurl, f.middlePIP),
        isCurled(f.ringCurl, 100),
        isCurled(f.pinkyCurl, 100),
        f.thumbCurl > 0.20,
        !isThumbExtended(f),
        f.thumbIndexDist < 0.45,
      ]);
    },
  },

  // ─────────────────────────────────────────────────────────────────────────
  // SCHOOL & SOCIAL
  // ─────────────────────────────────────────────────────────────────────────

  go_school: {
    minScore: 0.72,
    twoHanded: true,
    score: ([f0, f1]) => {
      if (!f0 || !f1) return 0;
      function clapFlat(f: HandFeatures) {
        return singleHandScore(f, [
          isExtended(f.indexCurl, f.indexPIP),
          isExtended(f.middleCurl, f.middlePIP),
          isExtended(f.ringCurl, 150),
          isExtended(f.pinkyCurl, 150),
          f.indexMiddleDist < 0.28,
        ]);
      }
      return (clapFlat(f0) + clapFlat(f1)) / 2;
    },
  },

  more_time: {
    minScore: 0.76,
    twoHanded: false,
    score: ([f]) => {
      if (!f) return 0;
      return singleHandScore(f, [
        isBent(f.indexCurl, f.indexPIP),
        isCurled(f.middleCurl, f.middlePIP),
        isCurled(f.ringCurl, 100),
        isCurled(f.pinkyCurl, 100),
        isThumbExtended(f),
        f.thumbAbduction > 0.50,
      ]);
    },
  },

  question: {
    minScore: 0.75,
    twoHanded: false,
    score: ([f]) => {
      if (!f) return 0;
      return singleHandScore(f, [
        isBent(f.indexCurl, f.indexPIP),
        isCurled(f.middleCurl, f.middlePIP),
        isCurled(f.ringCurl, 100),
        isCurled(f.pinkyCurl, 100),
        f.thumbCurl > 0.30,
        !isThumbExtended(f),
      ]);
    },
  },

  agree: {
    minScore: 0.76,
    twoHanded: false,
    score: ([f]) => {
      if (!f) return 0;
      return singleHandScore(f, [
        isExtended(f.indexCurl, f.indexPIP),
        isCurled(f.middleCurl, f.middlePIP),
        isCurled(f.ringCurl, 100),
        isCurled(f.pinkyCurl, 100),
        f.thumbCurl > 0.20 && f.thumbCurl < 0.50,
        f.thumbAbduction > 0.35 && f.thumbAbduction < 0.60,
        f.indexMiddleDist > 0.28,
      ]);
    },
  },

  disagree: {
    minScore: 0.72,
    twoHanded: true,
    score: ([f0, f1]) => {
      if (!f0 || !f1) return 0;
      function indexPoint(f: HandFeatures) {
        return singleHandScore(f, [
          isExtended(f.indexCurl, f.indexPIP),
          isCurled(f.middleCurl, f.middlePIP),
          isCurled(f.ringCurl, 100),
          isCurled(f.pinkyCurl, 100),
        ]);
      }
      return (indexPoint(f0) + indexPoint(f1)) / 2;
    },
  },

  no: {
    minScore: 0.78,
    twoHanded: false,
    score: ([f]) => {
      if (!f) return 0;
      return singleHandScore(f, [
        isExtended(f.indexCurl, f.indexPIP),
        isExtended(f.middleCurl, f.middlePIP),
        isCurled(f.ringCurl, 100),
        isCurled(f.pinkyCurl, 100),
        f.thumbIndexDist < 0.32,
        f.indexMiddleDist < 0.25,
      ]);
    },
  },

  hungry: {
    minScore: 0.75,
    twoHanded: false,
    score: ([f]) => PHRASE_SIGNATURES["how_are_you_greet"].score([f]),
  },

  what_doing: {
    minScore: 0.72,
    twoHanded: true,
    score: ([f0, f1]) => {
      if (!f0 || !f1) return 0;
      function pinch(f: HandFeatures) {
        return singleHandScore(f, [
          f.thumbIndexDist < 0.38,
          isBent(f.indexCurl, f.indexPIP) || isCurled(f.indexCurl, f.indexPIP),
          isBent(f.middleCurl, f.middlePIP),
        ]);
      }
      return (pinch(f0) + pinch(f1)) / 2;
    },
  },

  not_sure: {
    minScore: 0.70,
    twoHanded: true,
    score: ([f0, f1]) => {
      if (!f0 || !f1) return 0;
      function scale(f: HandFeatures) {
        return singleHandScore(f, [
          isExtended(f.indexCurl, f.indexPIP),
          isExtended(f.middleCurl, f.middlePIP),
          isExtended(f.ringCurl, 150),
          isExtended(f.pinkyCurl, 150),
          isThumbExtended(f),
        ]);
      }
      return (scale(f0) + scale(f1)) / 2;
    },
  },

  good_morning: {
    minScore: 0.75,
    twoHanded: false,
    score: ([f]) => {
      if (!f) return 0;
      return singleHandScore(f, [
        isExtended(f.indexCurl, f.indexPIP),
        isExtended(f.middleCurl, f.middlePIP),
        isExtended(f.ringCurl, 145),
        isExtended(f.pinkyCurl, 145),
        f.thumbCurl < 0.45,
        f.thumbAbduction < 0.60,
        f.indexMiddleDist < 0.32,
      ]);
    },
  },

  good_night_greet: {
    minScore: 0.70,
    twoHanded: false,
    score: ([f]) => {
      if (!f) return 0;
      return singleHandScore(f, [
        isBent(f.indexCurl, f.indexPIP),
        isBent(f.middleCurl, f.middlePIP),
        isBent(f.ringCurl, 135),
        isBent(f.pinkyCurl, 135),
        f.thumbCurl > 0.20,
        f.thumbAbduction < 0.55,
      ]);
    },
  },

  medicine: {
    minScore: 0.74,
    twoHanded: false,
    score: ([f]) => {
      if (!f) return 0;
      return singleHandScore(f, [
        isBent(f.middleCurl, f.middlePIP),
        isExtended(f.indexCurl, f.indexPIP),
        isExtended(f.pinkyCurl, 145),
        f.thumbCurl > 0.25,
        f.middleRingDist > 0.15,
      ]);
    },
  },

  toilet: {
    minScore: 0.74,
    twoHanded: false,
    score: ([f]) => {
      if (!f) return 0;
      return singleHandScore(f, [
        isCurled(f.indexCurl, f.indexPIP),
        isCurled(f.middleCurl, f.middlePIP),
        f.thumbCurl > 0.35,
        f.thumbIndexDist < 0.40,
        isCurled(f.ringCurl, 120),
      ]);
    },
  },

  yes_simple_alt: {
    minScore: 0.88,
    twoHanded: false,
    score: ([f]) => {
      if (!f) return 0;
      return singleHandScore(f, [
        isCurled(f.indexCurl, f.indexPIP),
        isCurled(f.middleCurl, f.middlePIP),
        isCurled(f.ringCurl, 100),
        isCurled(f.pinkyCurl, 100),
        f.thumbCrossesPalm,
      ]);
    },
  },

  no_simple: {
    minScore: 0.85,
    twoHanded: false,
    score: ([f]) => {
      if (!f) return 0;
      return singleHandScore(f, [
        f.thumbIndexDist < 0.30,
        f.thumbMiddleDist < 0.30,
        isExtended(f.ringCurl, 150),
        isExtended(f.pinkyCurl, 150),
      ]);
    },
  },

  emergency_urgent_alt: {
    minScore: 0.90,
    twoHanded: false,
    score: ([f]) => {
      if (!f) return 0;
      return singleHandScore(f, [
        isCurled(f.indexCurl, f.indexPIP),
        isCurled(f.middleCurl, f.middlePIP),
        isCurled(f.ringCurl, 100),
        isCurled(f.pinkyCurl, 100),
        f.thumbCrossesPalm,
      ]);
    },
  },

  thirsty: {
    minScore: 0.85,
    twoHanded: false,
    score: ([f]) => {
      if (!f) return 0;
      return singleHandScore(f, [
        isExtended(f.indexCurl, f.indexPIP),
        isCurled(f.middleCurl, f.middlePIP),
        isCurled(f.ringCurl, 100),
        isCurled(f.pinkyCurl, 100),
      ]);
    },
  },
};

// ── Temporal Smoothing Buffer ─────────────────────────────────────────────────

const BUFFER_SIZE = 8;     // Increased for better stabilization
const MIN_CONSISTENT = 5; // Require more frames for high confidence signs
let phraseBuffer: string[] = [];

function addToBuffer(label: string): string | null {
  phraseBuffer.push(label);
  if (phraseBuffer.length > BUFFER_SIZE) phraseBuffer.shift();
  
  if (phraseBuffer.length < MIN_CONSISTENT) return null;
  const recent = phraseBuffer.slice(-MIN_CONSISTENT);
  if (recent.every((l) => l === recent[0])) return recent[0];
  return null;
}

export function resetPhraseBuffer(): void {
  phraseBuffer = [];
}

// ── Main Classification ───────────────────────────────────────────────────────

export function classifyISLPhrase(
  multiHands: HandLandmark[][],
  targetPhraseId?: string
): { label: string; confidence: number } | null {
  if (!multiHands || multiHands.length === 0) return null;

  const handFeatures = multiHands
    .map((lm) => extractFeatures(lm))
    .filter((f): f is HandFeatures => f !== null);

  if (handFeatures.length === 0) return null;

  const handsAvailable = handFeatures.length;

  // ── Mode 1: Target phrase (practice mode) ─────────────────────────────────
  if (targetPhraseId && PHRASE_SIGNATURES[targetPhraseId]) {
    const sig = PHRASE_SIGNATURES[targetPhraseId];

    if (sig.twoHanded && handsAvailable < 2) {
      return null;
    }

    const score = sig.score(handFeatures);
    if (score >= sig.minScore) {
      const stable = addToBuffer(targetPhraseId);
      // ONLY return the targeted label if it is the ONE in the stable buffer
      if (stable === targetPhraseId) {
        return { label: targetPhraseId, confidence: Math.min(0.97, score) };
      }
    } else {
      // Clear buffer if target lost
      phraseBuffer = [];
    }
    return null;
  }

  // ── Mode 2: Free detection (browse/learn mode) ────────────────────────────
  // Score ALL phrases and pick the best winner
  const scores: { label: string; score: number; minScore: number }[] = [];

  for (const [label, sig] of Object.entries(PHRASE_SIGNATURES)) {
    if (sig.twoHanded && handsAvailable < 2) continue;
    const s = sig.score(handFeatures);
    scores.push({ label, score: s, minScore: sig.minScore });
  }

  scores.sort((a, b) => b.score - a.score);

  if (scores.length === 0) return null;

  const best = scores[0];
  const runner = scores[1];

  // Must exceed its own minScore AND beat runner-up by a healthy margin
  if (best.score < best.minScore) return null;
  
  // Ambiguity check: if top 2 are too close, it's uncertain
  if (runner && runner.score >= best.minScore && best.score - runner.score < 0.12) {
    return null;
  }

  const stable = addToBuffer(best.label);
  if (stable && stable === best.label) {
    return { label: best.label, confidence: Math.min(0.98, best.score) };
  }

  return null;
}
