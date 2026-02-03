import "../styles/globals.css";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "GEM Clinical ECG Workbench",
  description: "Clinical ECG diagnosis workspace",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <main>{children}</main>
      </body>
    </html>
  );
}
