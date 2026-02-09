interface EmptyStateProps {
  icon?: React.ReactNode;
  title: string;
  subtitle?: string;
  action?: React.ReactNode;
  className?: string;
}

const DEFAULT_ICON = (
  <svg className="w-10 h-10 text-fg-faint/50" fill="none" viewBox="0 0 24 24" stroke="currentColor">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M20 13V6a2 2 0 00-2-2H6a2 2 0 00-2 2v7m16 0v5a2 2 0 01-2 2H6a2 2 0 01-2-2v-5m16 0h-2.586a1 1 0 00-.707.293l-2.414 2.414a1 1 0 01-.707.293h-3.172a1 1 0 01-.707-.293l-2.414-2.414A1 1 0 006.586 13H4" />
  </svg>
);

export default function EmptyState({
  icon = DEFAULT_ICON,
  title,
  subtitle,
  action,
  className = "",
}: EmptyStateProps) {
  return (
    <div className={`flex flex-col items-center justify-center py-12 ${className}`}>
      <div className="mb-3 opacity-60">{icon}</div>
      <p className="text-fg-faint font-medium mb-1">{title}</p>
      {subtitle && <p className="text-fg-faint/70 text-sm max-w-xs text-center">{subtitle}</p>}
      {action && <div className="mt-4">{action}</div>}
    </div>
  );
}
