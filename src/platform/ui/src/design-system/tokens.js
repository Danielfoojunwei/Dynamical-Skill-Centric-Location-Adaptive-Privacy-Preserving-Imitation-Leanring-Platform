/**
 * Enterprise Design System Tokens
 *
 * Unified design tokens following enterprise patterns:
 * - Consistent spacing scale
 * - Semantic color system
 * - Typography hierarchy
 * - Shadow/elevation system
 *
 * @version 1.0.0
 */

export const colors = {
  // Brand colors
  brand: {
    primary: '#3b82f6',
    secondary: '#8b5cf6',
    accent: '#06b6d4',
  },

  // Semantic colors
  semantic: {
    success: '#22c55e',
    warning: '#f59e0b',
    error: '#ef4444',
    info: '#3b82f6',
  },

  // Status colors
  status: {
    online: '#22c55e',
    offline: '#6b7280',
    degraded: '#f59e0b',
    critical: '#ef4444',
    maintenance: '#8b5cf6',
  },

  // Background colors
  bg: {
    primary: '#0f172a',
    secondary: '#1e293b',
    tertiary: '#334155',
    elevated: '#1e293b',
    overlay: 'rgba(0, 0, 0, 0.75)',
  },

  // Text colors
  text: {
    primary: '#f8fafc',
    secondary: '#94a3b8',
    muted: '#64748b',
    inverse: '#0f172a',
  },

  // Border colors
  border: {
    default: '#334155',
    subtle: '#1e293b',
    strong: '#475569',
  },
};

export const spacing = {
  xs: '0.25rem',   // 4px
  sm: '0.5rem',    // 8px
  md: '1rem',      // 16px
  lg: '1.5rem',    // 24px
  xl: '2rem',      // 32px
  '2xl': '3rem',   // 48px
  '3xl': '4rem',   // 64px
};

export const typography = {
  fontFamily: {
    sans: "'Inter', system-ui, -apple-system, sans-serif",
    mono: "'JetBrains Mono', 'Fira Code', monospace",
  },
  fontSize: {
    xs: '0.75rem',
    sm: '0.875rem',
    md: '1rem',
    lg: '1.125rem',
    xl: '1.25rem',
    '2xl': '1.5rem',
    '3xl': '2rem',
    '4xl': '2.5rem',
  },
  fontWeight: {
    normal: 400,
    medium: 500,
    semibold: 600,
    bold: 700,
  },
  lineHeight: {
    tight: 1.25,
    normal: 1.5,
    relaxed: 1.75,
  },
};

export const shadows = {
  sm: '0 1px 2px 0 rgba(0, 0, 0, 0.05)',
  md: '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
  lg: '0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)',
  xl: '0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04)',
  glow: {
    blue: '0 0 20px rgba(59, 130, 246, 0.3)',
    green: '0 0 20px rgba(34, 197, 94, 0.3)',
    red: '0 0 20px rgba(239, 68, 68, 0.3)',
    purple: '0 0 20px rgba(139, 92, 246, 0.3)',
  },
};

export const radii = {
  sm: '0.25rem',
  md: '0.5rem',
  lg: '0.75rem',
  xl: '1rem',
  full: '9999px',
};

export const transitions = {
  fast: '150ms ease',
  normal: '200ms ease',
  slow: '300ms ease',
};

export const breakpoints = {
  sm: '640px',
  md: '768px',
  lg: '1024px',
  xl: '1280px',
  '2xl': '1536px',
};

// Z-index scale
export const zIndex = {
  base: 0,
  dropdown: 100,
  sticky: 200,
  modal: 300,
  popover: 400,
  tooltip: 500,
  toast: 600,
};

export default {
  colors,
  spacing,
  typography,
  shadows,
  radii,
  transitions,
  breakpoints,
  zIndex,
};
