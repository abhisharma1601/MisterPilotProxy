# MisterPilot

An AI coding assistant that lives in your VS Code sidebar. Ask questions, get code written, and review diffs — all without leaving your editor.

Powered by **DeepSeek** — a state-of-the-art AI model that matches the quality of leading coding assistants at a fraction of the cost, keeping MisterPilot accessible and affordable for everyone.

## Features

- **Chat interface** — conversational AI directly in the sidebar
- **File operations** — reads, writes, and replaces file content on your local machine
- **Diff review** — proposed edits are shown as diffs for you to approve or reject before anything is written
- **Terminal commands** — the agent can suggest terminal commands; you approve before they run
- **Code search** — searches across your workspace using ripgrep

## Why DeepSeek?

Most AI coding tools rely on expensive models that drive up subscription costs. MisterPilot uses DeepSeek — a highly capable open-weight model that delivers excellent coding performance at significantly lower cost. This means:

- No expensive monthly subscription
- Fast responses without cutting corners on quality
- A sustainable model that doesn't lock you into big-tech pricing

## Getting Started

### Step 1 — Get your DeepSeek API key

1. Go to [platform.deepseek.com](https://platform.deepseek.com) and create an account
2. Navigate to **API Keys** in your dashboard and generate a new key
3. Copy the key — you'll need it in the next step

### Step 2 — Install and configure the extension

1. Install MisterPilot from the VS Code Marketplace
2. Open the MisterPilot panel from the activity bar (look for the icon on the left)
3. Open the command palette with `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac)
4. Run **MisterPilot: Set API Key** and paste your key
5. Start chatting

That's it — no other configuration needed.

## How It Works

MisterPilot sends your messages to the backend at `misterpilot.online` which drives the AI. All file system and terminal operations execute **locally in your VS Code extension** — nothing on your machine is touched without your explicit approval.

## Privacy & Data Security

MisterPilot is built with your privacy in mind. Before any data leaves your machine:

- All messages and file content are scanned and **PII (Personally Identifiable Information) is filtered out** — including emails, phone numbers, credentials, and other sensitive identifiers
- No raw personal data is stored or logged on our servers
- Only the content needed to answer your query is transmitted
- **DeepSeek never sees your actual data** — all content is filtered and anonymised by MisterPilot before it reaches the model

## Documentation

Full documentation, guides, and API reference at **[misterpilot.online/docs](https://misterpilot.online/docs)**

## Requirements

- VS Code 1.85 or newer
- A DeepSeek API key (get one at [platform.deepseek.com](https://platform.deepseek.com))
