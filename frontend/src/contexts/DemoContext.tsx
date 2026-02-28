import { createContext, useContext, useEffect, useState, ReactNode } from "react";
import { checkBackendHealth, isDemoMode } from "../services/apiClient";

interface DemoContextValue {
    isDemo: boolean;
    loading: boolean;
}

const DemoContext = createContext<DemoContextValue>({ isDemo: false, loading: true });

export function useDemoMode() {
    return useContext(DemoContext);
}

export function DemoProvider({ children }: { children: ReactNode }) {
    const [isDemo, setIsDemo] = useState(false);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        checkBackendHealth().then(() => {
            setIsDemo(isDemoMode());
            setLoading(false);
        });
    }, []);

    return (
        <DemoContext.Provider value={{ isDemo, loading }}>
            {children}
        </DemoContext.Provider>
    );
}
