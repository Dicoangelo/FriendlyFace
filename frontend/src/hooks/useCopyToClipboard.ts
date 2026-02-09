import { useToast } from "./useToast";

export function useCopyToClipboard() {
  const toast = useToast();

  return (text: string, label = "Copied to clipboard") => {
    navigator.clipboard.writeText(text).then(
      () => toast.info(label),
      () => toast.error("Failed to copy"),
    );
  };
}
