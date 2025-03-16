import { useState, useEffect, useRef } from "react";
import axios from "axios";
import { Send, MessageCircle, X } from "lucide-react";

const Chatbot = () => {
    const initialMessages = [
        {
            sender: "bot",
            text: "Hello there! 👋 Welcome to TripDarzee! I'm TravelMitra, your personal travel assistant. What kind of trip are you dreaming of? Tell me all about it, and I'll help you weave a perfect travel tapestry! 🧵✨"
        }
    ];

    const [messages, setMessages] = useState(initialMessages);
    const [input, setInput] = useState("");
    const [isOpen, setIsOpen] = useState(false); 
    const [followUpQuestions, setFollowUpQuestions] = useState([]);
    const chatEndRef = useRef(null);

    const scrollToBottom = () => {
        chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    const toggleChatbot = () => {
        if (isOpen) {
            setMessages(initialMessages);
            setFollowUpQuestions([]);
        }
        setIsOpen(!isOpen);
    };

    const sendMessage = async (userMessage) => {
        if (userMessage.trim() === "") return;

        setMessages(prevMessages => [...prevMessages, { sender: "user", text: userMessage }]);
        setInput("");

        try {
            const response = await axios.post("http://localhost:5000/chat", {
                message: userMessage,
            });

            const botMessage = response.data.response;
            setMessages(prevMessages => [...prevMessages, { sender: "bot", text: botMessage }]);

            if (response.data.followUpQuestions) {
                setFollowUpQuestions(response.data.followUpQuestions);
            }
        } catch (error) {
            setMessages(prevMessages => [...prevMessages, {
                sender: "bot",
                text: "Oops! Something went wrong. Please try again later."
            }]);
        }
    };

    const handleSendMessage = (e) => {
        e.preventDefault();
        sendMessage(input);
    };

    const handleFollowUpClick = (question) => {
        sendMessage(question);
    };

    return (
        <>
            {!isOpen && (
                <button 
                    onClick={toggleChatbot}
                    className="fixed bottom-10 right-10 bg-blue-600 text-white p-4 rounded-full shadow-lg hover:bg-blue-700 transition-transform transform hover:scale-110 duration-300"
                >
                    <MessageCircle size={30} />
                </button>
            )}

            {isOpen && (
                <div className="fixed bottom-10 right-10 w-full max-w-sm bg-white shadow-xl rounded-2xl overflow-hidden border transform transition-transform duration-300">
                    <div className="bg-blue-600 text-white flex justify-between items-center p-4 rounded-t-2xl">
                        <span className="font-semibold">TravelMitra - TripDarzee Chat</span>
                        <button onClick={toggleChatbot} className="text-white hover:text-gray-300">
                            <X size={24} />
                        </button>
                    </div>
                    <div className="h-80 overflow-y-auto p-4 space-y-4">
                        {messages.map((msg, index) => (
                            <div
                                key={index}
                                className={`flex ${msg.sender === "user" ? "justify-end" : "justify-start"}`}
                            >
                                <div
                                    className={`p-3 rounded-2xl max-w-xs shadow-sm ${
                                        msg.sender === "user" 
                                            ? "bg-blue-500 text-white" 
                                            : "bg-gray-200 text-black"
                                    }`}
                                >
                                    {msg.text}
                                </div>
                            </div>
                        ))}
                        <div ref={chatEndRef}></div>
                    </div>
                    <form onSubmit={handleSendMessage} className="flex items-center p-2 border-t">
                        <input
                            type="text"
                            value={input}
                            onChange={(e) => setInput(e.target.value)}
                            placeholder="Type a message..."
                            className="w-full p-2 text-gray-700 border rounded-full focus:outline-none focus:ring-2 focus:ring-blue-500"
                        />
                        <button type="submit" className="ml-2 text-blue-600 hover:text-blue-800">
                            <Send size={24} />
                        </button>
                    </form>
                    {followUpQuestions.length > 0 && (
                        <div className="p-4 border-t bg-gray-50">
                            <p className="text-gray-600 mb-2">Quick Questions:</p>
                            <div className="flex flex-wrap gap-2">
                                {followUpQuestions.map((question, index) => (
                                    <button 
                                        key={index}
                                        onClick={() => handleFollowUpClick(question)}
                                        className="bg-blue-100 text-blue-600 px-3 py-1 rounded-full text-sm hover:bg-blue-200 transition duration-200"
                                    >
                                        {question}
                                    </button>
                                ))}
                            </div>
                        </div>
                    )}
                </div>
            )}
        </>
    );
};

export default Chatbot;
