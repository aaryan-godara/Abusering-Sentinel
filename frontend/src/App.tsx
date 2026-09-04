import { Routes, Route, useNavigate, useLocation } from 'react-router-dom';
import { useState } from 'react';
import { Search, Crosshair } from 'lucide-react';
import Dashboard from './pages/Dashboard';
import Investigation from './pages/Investigation';

/* ==========================================================================
   AbuseRing Sentinel — console shell
   Fixed header, then a single full-bleed workspace surface. No page padding:
   the graph owns the viewport and every panel floats above it.
   ========================================================================== */

export default function App() {
  const [search, setSearch] = useState('');
  const navigate = useNavigate();
  const location = useLocation();

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    const id = search.trim();
    if (id) navigate(`/users/${id}`);
  };

  const onInvestigation = location.pathname.startsWith('/users/');

  return (
    <div className="tech-grid h-full flex flex-col text-ink-100 font-sans overflow-hidden">
      <header className="floating-panel-strong relative z-40 mx-2 mt-2 flex h-11 shrink-0 items-center gap-4 px-3">
        <button
          type="button"
          onClick={() => navigate('/')}
          className="flex items-center gap-2 text-left transition-colors hover:text-signal-bright"
        >
          <Crosshair className="h-4 w-4 text-signal" strokeWidth={1.8} />
          <span className="ui-heading uppercase tracking-[0.08em]">
            AbuseRing <span className="text-signal">Sentinel</span>
          </span>
        </button>

        <span className="label-caps hidden sm:inline">
          / {onInvestigation ? 'Investigation' : 'Workspace'}
        </span>

        <form onSubmit={handleSearch} className="ml-auto relative w-56 sm:w-80">
          <Search
            className="pointer-events-none absolute left-2 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-ink-350"
            strokeWidth={1.8}
          />
          <input
            type="text"
            className="tech-id w-full rounded-md border border-ink-600 bg-white/70 py-[7px] pl-7 pr-3 text-[12.5px] font-medium text-ink-50 placeholder-ink-350 backdrop-blur transition-colors focus:border-signal focus:outline-none"
            placeholder="USER ID — e.g. USR_00000085"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            aria-label="Search user id"
          />
        </form>
      </header>

      <main className="relative min-h-0 flex-1 overflow-hidden">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/users/:userId" element={<Investigation />} />
        </Routes>
      </main>
    </div>
  );
}
