"use client";
// components/VisionContainer.tsx
import React, { useState, useCallback } from "react";
import { useControlContext } from "@/providers/ControlContext";
import { Card } from "@/components/ui/card";

import { MarkdownViewer } from "./markdown-viewer/MarkdownViewer";
import { CommonForm } from "./CommonForm";
import { TypingBubble } from "./TypingBubble";

import { RefreshCcw } from "lucide-react";
import { Button } from "./ui/button";

export const VisionContainer = () => {
  const { generalSettings, safetySettings, mediaDataList } = useControlContext();

  const [chatHistory, setChatHistory] = useState<
    { prompt: string; response: string; image?: string }[]
  >([]);
  const [prompt, setPrompt] = useState<string>("");
  const [loading, setLoading] = useState<boolean>(false);

  const isFormSubmittable = useCallback(() => {
    return (
      prompt.trim() !== "" &&
      mediaDataList.some(
        (media) => media !== null && media.data !== "" && media.mimeType !== ""
      )
    );
  }, [prompt, mediaDataList]);

  const makeApiCall = useCallback(
    async (message: string) => {
      console.log("TEST");
      // Filter out any invalid image data
      const validMediaData = mediaDataList.filter(
        (data) => data.data !== "" && data.mimeType !== ""
      );

      // If there are no valid images and the message is empty, do not proceed
      if (validMediaData.length === 0) return;
      if (message.trim() === "") return;

      const mediaBase64 = validMediaData.map((data) =>
        data.data.replace(/^data:(image|video)\/\w+;base64,/, "")
      );
      const mediaTypes = validMediaData.map((data) => data.mimeType);

      const body = JSON.stringify({
        message,
        media: mediaBase64,
        media_types: mediaTypes,
        general_settings: generalSettings,
        safety_settings: safetySettings,
      });

      try {
        const response = await fetch(`http://127.0.0.1:8000/run-image`, {
          method: "POST",
          body,
          headers: {
            "Content-Type": "application/json",
          },
        });

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        const responseData = await response.json();
        const { status, prompt_result, image } = responseData;
        console.log("response data")

        setChatHistory((prevChatHistory) => [
          ...prevChatHistory,
          { prompt: message, response: prompt_result, image: image },
        ]);
      } catch (error) {
        if (error instanceof Error) {
          setChatHistory((prevChatHistory) => [
            ...prevChatHistory,
            { prompt: message, response: `Error: ${error.message}` },
          ]);
        }
      } finally {
        setLoading(false);
      }
    },
    [mediaDataList, generalSettings, safetySettings]
  );

  const handleSubmitForm = useCallback(
    async (e: React.FormEvent<HTMLFormElement>) => {
      e.preventDefault();

      if (!isFormSubmittable()) return;
      setLoading(true);
      const currentPrompt = prompt;
      setPrompt("");
      await makeApiCall(currentPrompt);
    },
    [isFormSubmittable, makeApiCall, prompt]
  );

  return (
    <div className="flex flex-col h-[95vh]">
      <Card className="flex flex-col flex-1 overflow-hidden">
        <div className="flex-1 overflow-y-auto p-4">
          {chatHistory.map((chat, index) => (
            <div key={index} className="mb-4">
              <div className="bg-gray-200 p-2 rounded">
                <strong>User:</strong> {chat.prompt}
              </div>
              <div className="bg-gray-100 p-2 rounded mt-1">
                <strong>Response:</strong>
                <MarkdownViewer text={chat.response} />
                {chat.image && (
                  <img
                    src={`data:image/png;base64,${chat.image}`}
                    alt="Generated result"
                    className="mt-2"
                  />
                )}
              </div>
            </div>
          ))}
          {loading && <TypingBubble />}
          {mediaDataList.every((media) => media === null || media?.data === "") && (
            <div className="flex flex-col items-center justify-center h-full">
              <div className="text-2xl text-primary/80 font-medium">
                Add an image to get started
              </div>
            </div>
          )}
        </div>
        <CommonForm
          value={prompt}
          placeholder="Ask a question about the images"
          loading={loading}
          onInputChange={(e) => setPrompt(e.target.value)}
          onFormSubmit={handleSubmitForm}
          isSubmittable={isFormSubmittable()}
        />
      </Card>
    </div>
  );
};
