interface ProgressRingProps {
  value: number;        // 0-1
  size?: number;        // px
  strokeWidth?: number; // px
  color?: string;       // tailwind color class
  label?: string;
  className?: string;
}

export default function ProgressRing({
  value,
  size = 64,
  strokeWidth = 5,
  color = "text-cyan",
  label,
  className = "",
}: ProgressRingProps) {
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference * (1 - Math.min(Math.max(value, 0), 1));

  return (
    <div className={`relative inline-flex items-center justify-center ${className}`} style={{ width: size, height: size }}>
      <svg width={size} height={size} className="-rotate-90">
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke="currentColor"
          strokeWidth={strokeWidth}
          className="text-fg-faint/20"
        />
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke="currentColor"
          strokeWidth={strokeWidth}
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          strokeLinecap="round"
          className={`${color} transition-all duration-700 ease-out`}
        />
      </svg>
      {label && (
        <span className={`absolute text-xs font-bold ${color}`}>{label}</span>
      )}
    </div>
  );
}
