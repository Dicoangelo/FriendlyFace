import { useContext } from "react";
import { ToastContext } from "../contexts/ToastContext";
import type { ToastActions } from "../contexts/ToastContext";

export function useToast(): ToastActions {
  const ctx = useContext(ToastContext);
  if (!ctx) {
    throw new Error("useToast must be used within a <ToastProvider>");
  }
  return ctx;
}
