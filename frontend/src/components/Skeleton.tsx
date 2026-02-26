export function SkeletonCard({ className = "h-24" }: { className?: string }) {
  return <div className={`skeleton-shimmer rounded-xl ${className}`} />;
}

export function SkeletonRow() {
  return (
    <div className="flex items-center gap-4 py-3 border-b border-border-theme">
      <div className="w-20 h-4 skeleton-shimmer rounded" />
      <div className="w-16 h-5 skeleton-shimmer rounded-full" />
      <div className="flex-1 h-4 skeleton-shimmer rounded" />
      <div className="w-32 h-4 skeleton-shimmer rounded" />
    </div>
  );
}

export function SkeletonTable({ rows = 5 }: { rows?: number }) {
  return (
    <div className="glass-card p-4">
      <div className="w-40 h-5 skeleton-shimmer rounded mb-4" />
      {[...Array(rows)].map((_, i) => (
        <SkeletonRow key={i} />
      ))}
    </div>
  );
}

export function SkeletonDashboard() {
  return (
    <div className="space-y-6 animate-fade-in">
      {/* Quick actions skeleton */}
      <div className="glass-card p-4">
        <div className="flex items-center justify-between">
          <div className="space-y-2">
            <div className="w-32 h-5 skeleton-shimmer rounded" />
            <div className="w-48 h-4 skeleton-shimmer rounded" />
          </div>
          <div className="flex gap-2">
            {[...Array(4)].map((_, i) => (
              <div key={i} className="w-24 h-8 skeleton-shimmer rounded-lg" />
            ))}
          </div>
        </div>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {[...Array(6)].map((_, i) => (
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
