"use client"

import { useState, useEffect } from "react"

export function useLocalStorage<T>(key: string, initialValue: T) {
  const [value, setValue] = useState<T>(initialValue)
  const [hydrated, setHydrated] = useState(false)

  useEffect(() => {
    try {
      const stored = localStorage.getItem(key)
      if (stored !== null) {
        setValue(JSON.parse(stored) as T)
      }
    } catch {
      // ignore
    }
    setHydrated(true)
  }, [key])

  function set(next: T | ((prev: T) => T)) {
    setValue(prev => {
      const resolved = typeof next === "function" ? (next as (prev: T) => T)(prev) : next
      try {
        localStorage.setItem(key, JSON.stringify(resolved))
      } catch {
        // ignore
      }
      return resolved
    })
  }

  return [value, set, hydrated] as const
}
