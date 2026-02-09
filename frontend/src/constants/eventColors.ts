/**
 * Unified event type color system.
 *
 * Color families:
 *   amethyst — training, models, bundles (AI/ML operations)
 *   cyan     — inference, FL rounds (real-time pipeline)
 *   teal     — explanations, consent (trust & governance)
 *   gold     — audits, compliance (review & oversight)
 *   rose-ember — security alerts (critical)
 */

type ColorFamily = "amethyst" | "cyan" | "teal" | "gold" | "rose-ember";

const COLOR_FAMILIES: Record<ColorFamily, { badge: string; border: string; bar: { bg: string; text: string } }> = {
  amethyst: {
    badge: "bg-amethyst/10 text-amethyst",
    border: "border-amethyst/30",
    bar: { bg: "bg-amethyst/30", text: "text-amethyst" },
  },
  cyan: {
    badge: "bg-cyan/10 text-cyan",
    border: "border-cyan/30",
    bar: { bg: "bg-cyan/30", text: "text-cyan" },
  },
  teal: {
    badge: "bg-teal/10 text-teal",
    border: "border-teal/30",
    bar: { bg: "bg-teal/30", text: "text-teal" },
  },
  gold: {
    badge: "bg-gold/10 text-gold",
    border: "border-gold/30",
    bar: { bg: "bg-gold/30", text: "text-gold" },
  },
  "rose-ember": {
    badge: "bg-rose-ember/10 text-rose-ember",
    border: "border-rose-ember/30",
    bar: { bg: "bg-rose-ember/30", text: "text-rose-ember" },
  },
};

const EVENT_FAMILY: Record<string, ColorFamily> = {
  training_start: "amethyst",
  training_complete: "amethyst",
  model_registered: "amethyst",
  inference_request: "cyan",
  inference_result: "cyan",
  explanation_generated: "teal",
  bias_audit: "gold",
  consent_recorded: "teal",
  consent_update: "teal",
  bundle_created: "amethyst",
  fl_round: "cyan",
  security_alert: "rose-ember",
  compliance_report: "gold",
};

const DEFAULT_FAMILY = COLOR_FAMILIES.cyan;

function getFamily(type: string) {
  return COLOR_FAMILIES[EVENT_FAMILY[type]] || DEFAULT_FAMILY;
}

/** Badge classes: bg + text (no border) */
export function eventBadgeColor(type: string): string {
  return getFamily(type).badge;
}

/** Full badge classes: bg + text + border */
export function eventTypeColor(type: string): string {
  const f = getFamily(type);
  return `${f.badge} ${f.border}`;
}

/** Border-only class */
export function eventBorderColor(type: string): string {
  return getFamily(type).border;
}

/** Bar chart colors: { bg, text } */
export function eventBarColor(type: string): { bg: string; text: string } {
  return getFamily(type).bar;
}

/** Status badge colors for pass/warning/fail/unknown */
export const STATUS_COLORS: Record<string, string> = {
  pass: "bg-teal/10 text-teal border-teal/20",
  warning: "bg-gold/10 text-gold border-gold/20",
  fail: "bg-rose-ember/10 text-rose-ember border-rose-ember/20",
  unknown: "bg-fg/5 text-fg-muted border-border-theme",
};

export function statusColor(status: string): string {
  return STATUS_COLORS[status] || STATUS_COLORS.unknown;
}
