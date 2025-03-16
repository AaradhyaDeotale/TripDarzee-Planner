import { initializeApp } from "firebase/app";
import { getFirestore } from "firebase/firestore";
import { getStorage, connectStorageEmulator } from "firebase/storage"; // Add this line

// Your web app's Firebase configuration
const firebaseConfig = {
  apiKey: import.meta.env.VITE_FIREBASE_API_KEY,
  authDomain: import.meta.env.VITE_FIREBASE_AUTH_DOMAIN,
  projectId: import.meta.env.VITE_FIREBASE_PROJECT_ID,
  storageBucket: import.meta.env.VITE_FIREBASE_STORAGE_BUCKET,
  messagingSenderId: import.meta.env.VITE_FIREBASE_MESSAGING_SENDER_ID,
  appId: import.meta.env.VITE_FIREBASE_APP_ID,
};

// Initialize Firebase
export const app = initializeApp(firebaseConfig);

// Initialize Firestore
export const db = getFirestore(app);

// Initialize Firebase Storage
export const storage = getStorage(app); // Add this line

// Connect to the Storage Emulator if running locally
if (window.location.hostname === "localhost") {
  connectStorageEmulator(storage, "127.0.0.1", 4000); // Add this line
}