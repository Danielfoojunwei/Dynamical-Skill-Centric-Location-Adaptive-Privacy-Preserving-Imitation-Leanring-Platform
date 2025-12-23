/**
 * Toast Notification Component
 *
 * Displays temporary notification messages with auto-dismiss.
 * Uses Radix UI Toast for accessibility.
 */

import React from 'react';
import * as ToastPrimitive from '@radix-ui/react-toast';
import { X, CheckCircle, AlertCircle, AlertTriangle, Info } from 'lucide-react';
import { useUiStore } from '../../stores';
import { clsx } from 'clsx';

const icons = {
  success: CheckCircle,
  error: AlertCircle,
  warning: AlertTriangle,
  info: Info,
};

const styles = {
  success: 'bg-green-900/90 border-green-500 text-green-100',
  error: 'bg-red-900/90 border-red-500 text-red-100',
  warning: 'bg-yellow-900/90 border-yellow-500 text-yellow-100',
  info: 'bg-blue-900/90 border-blue-500 text-blue-100',
};

const iconStyles = {
  success: 'text-green-400',
  error: 'text-red-400',
  warning: 'text-yellow-400',
  info: 'text-blue-400',
};

function Toast({ toast, onClose }) {
  const Icon = icons[toast.type] || Info;

  return (
    <ToastPrimitive.Root
      className={clsx(
        'flex items-start gap-3 p-4 rounded-lg border shadow-lg',
        'data-[state=open]:animate-slideIn data-[state=closed]:animate-slideOut',
        'data-[swipe=end]:animate-swipeOut',
        styles[toast.type]
      )}
      duration={toast.duration}
      onOpenChange={(open) => !open && onClose()}
    >
      <Icon className={clsx('w-5 h-5 mt-0.5 flex-shrink-0', iconStyles[toast.type])} />

      <div className="flex-1 min-w-0">
        {toast.title && (
          <ToastPrimitive.Title className="font-semibold text-sm">
            {toast.title}
          </ToastPrimitive.Title>
        )}
        {toast.message && (
          <ToastPrimitive.Description className="text-sm opacity-90 mt-1">
            {toast.message}
          </ToastPrimitive.Description>
        )}
      </div>

      <ToastPrimitive.Close
        className="flex-shrink-0 p-1 rounded hover:bg-white/10 transition-colors"
        aria-label="Close"
      >
        <X className="w-4 h-4" />
      </ToastPrimitive.Close>
    </ToastPrimitive.Root>
  );
}

export function ToastProvider({ children }) {
  const toasts = useUiStore((state) => state.toasts);
  const removeToast = useUiStore((state) => state.removeToast);

  return (
    <ToastPrimitive.Provider swipeDirection="right">
      {children}

      {toasts.map((toast) => (
        <Toast
          key={toast.id}
          toast={toast}
          onClose={() => removeToast(toast.id)}
        />
      ))}

      <ToastPrimitive.Viewport
        className={clsx(
          'fixed bottom-4 right-4 flex flex-col gap-2 w-96 max-w-[calc(100vw-2rem)]',
          'z-50 outline-none'
        )}
      />
    </ToastPrimitive.Provider>
  );
}

export default Toast;
