export function SkeletonCard({ className = "h-24" }: { className?: string }) {
  return <div className={`bg-surface rounded-lg animate-pulse ${className}`} />;
}

export function SkeletonRow() {
  return (
    <div className="flex items-center gap-4 py-3 border-b border-border-theme">
      <div className="w-20 h-4 bg-surface rounded animate-pulse" />
      <div className="w-16 h-5 bg-surface rounded-full animate-pulse" />
      <div className="flex-1 h-4 bg-surface rounded animate-pulse" />
      <div className="w-32 h-4 bg-surface rounded animate-pulse" />
    </div>
  );
}

export function SkeletonTable({ rows = 5 }: { rows?: number }) {
  return (
    <div className="glass-card p-4">
      <div className="w-40 h-5 bg-surface rounded animate-pulse mb-4" />
      {[...Array(rows)].map((_, i) => (
        <SkeletonRow key={i} />
      ))}
    </div>
  );
}

export function SkeletonDashboard() {
  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {[...Array(4)].map((_, i) => (
          <SkeletonCard key={i} />
        ))}
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <SkeletonCard className="h-32" />
        <SkeletonCard className="h-32" />
      </div>
      <SkeletonCard className="h-48" />
    </div>
  );
}
