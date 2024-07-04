"use client";
// components/CommonForm.tsx
import React, {
  useState,
  useRef,
  useCallback,
  FormEvent,
  KeyboardEvent,
  useEffect,
} from "react";
import { Button } from "./ui/button";
import { Loader, Send } from "lucide-react";

interface CommonFormProps {
  value: string;
  placeholder: string;
  loading: boolean;
  onInputChange: (event: React.ChangeEvent<HTMLTextAreaElement>) => void;
  onFormSubmit: (event: FormEvent<HTMLFormElement>) => void;
  isSubmittable: boolean;
}

export const CommonForm: React.FC<CommonFormProps> = ({
  value,
  placeholder,
  loading,
  onInputChange,
  onFormSubmit,
  isSubmittable,
}) => {
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const [textareaHeight, setTextareaHeight] = useState("auto");

  const handleKeyPress = useCallback(
    (event: KeyboardEvent<HTMLTextAreaElement>) => {
      if (event.key === "Enter" && !event.ctrlKey && !event.shiftKey) {
        event.preventDefault();
        if (isSubmittable && textareaRef.current) {
          onFormSubmit(event as unknown as FormEvent<HTMLFormElement>);
        }
      } else if (event.key === "Enter") {
        console.log("test2")

        event.preventDefault();
        const textarea = event.currentTarget;
        const cursorPosition = textarea.selectionStart;
        const newValue =
          textarea.value.slice(0, cursorPosition) +
          "\n" +
          textarea.value.slice(cursorPosition);
        textarea.value = newValue;
        const changeEvent = new Event("input", {
          bubbles: true,
        }) as unknown as React.ChangeEvent<HTMLTextAreaElement>;
        Object.defineProperty(changeEvent, "target", {
          writable: true,
          value: { value: newValue },
        });
        onInputChange(changeEvent);
        textarea.selectionStart = cursorPosition + 1;
        textarea.selectionEnd = cursorPosition + 1;
        adjustTextareaHeight(textarea);
      }
    },
    [onInputChange, onFormSubmit, isSubmittable]
  );

  const handleTextAreaInput = useCallback(
    (event: React.ChangeEvent<HTMLTextAreaElement>) => {
      onInputChange(event);
      adjustTextareaHeight(event.currentTarget);
    },
    [onInputChange]
  );

  const resetTextareaHeight = useCallback(() => {
    setTextareaHeight("2.5rem");
  }, []);

  const adjustTextareaHeight = (target: HTMLTextAreaElement) => {
    target.style.height = "auto";
    target.style.height = `${target.scrollHeight}px`;
  };

  useEffect(() => {
    if (loading) resetTextareaHeight();
  }, [loading, resetTextareaHeight]);

  return (
    <form
    
      onSubmit={onFormSubmit}
      className="flex w-full items-center   border-t border-primary/70 pb-4 pt-6"
    >

      <label htmlFor="prompt" className="sr-only">Enter your prompt</label>
      <div>
        <button
          className="hover:text-blue-600 dark:text-slate-200 dark:hover:text-blue-600 sm:p-2"
          type="button"
        >
          <svg
            xmlns="http://www.w3.org/2000/svg"
            className="h-6 w-6"
            aria-hidden="true"
            viewBox="0 0 24 24"
            strokeWidth="2"
            stroke="currentColor"
            fill="none"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <path stroke="none" d="M0 0h24v24H0z" fill="none"></path>
            <path d="M12 5l0 14"></path>
            <path d="M5 12l14 0"></path>
          </svg>
          <span className="sr-only">Attach file</span>
        </button>
      </div>
      <textarea
        id="prompt"
        ref={textareaRef}
        value={value}
        onInput={handleTextAreaInput}
        onChange={onInputChange}
        onKeyDown={handleKeyPress}
        style={{ height: textareaHeight }}
        rows={1}
        className="flex-1 p-2 resize-none min-h-8 rounded max-h-[50vh]"
        placeholder={placeholder}
      />
      <div>
        <button
          className="inline-flex hover:text-blue-600 dark:text-slate-200 dark:hover:text-blue-600 sm:p-2"
          type="submit"
          disabled={!isSubmittable || loading}
        >
          {loading ? (
            <Loader className="animate-spin h-6 w-6" />
          ) : (
            <svg
              xmlns="http://www.w3.org/2000/svg"
              className="h-6 w-6"
              aria-hidden="true"
              viewBox="0 0 24 24"
              strokeWidth="2"
              stroke="currentColor"
              fill="none"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <path stroke="none" d="M0 0h24v24H0z" fill="none"></path>
              <path d="M10 14l11 -11"></path>
              <path
                d="M21 3l-6.5 18a.55 .55 0 0 1 -1 0l-3.5 -7l-7 -3.5a.55 .55 0 0 1 0 -1l18 -6.5"
              ></path>
            </svg>
          )}
          <span className="sr-only">Send message</span>
        </button>
      </div>
    </form>
  );
};
