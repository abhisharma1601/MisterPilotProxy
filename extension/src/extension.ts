import * as vscode from 'vscode';
import { SidebarProvider } from './sidebarProvider';

export function activate(context: vscode.ExtensionContext): void {
  const provider = new SidebarProvider(context);

  context.subscriptions.push(
    vscode.window.registerWebviewViewProvider(
      SidebarProvider.viewType,
      provider,
      { webviewOptions: { retainContextWhenHidden: true } }
    )
  );

  context.subscriptions.push(
    vscode.commands.registerCommand('misterpilot.setApiKey', async () => {
      const key = await vscode.window.showInputBox({
        prompt: 'Enter your API key for the LLM backend',
        placeHolder: 'sk-...',
        password: true,
        ignoreFocusOut: true,
      });
      if (key === undefined) return;
      if (key === '') {
        await context.secrets.delete('misterpilot.apiKey');
        vscode.window.showInformationMessage('MisterPilot: API key cleared.');
      } else {
        await context.secrets.store('misterpilot.apiKey', key);
        vscode.window.showInformationMessage('MisterPilot: API key saved.');
      }
    })
  );

  context.subscriptions.push(
    vscode.commands.registerCommand('misterpilot.setBackendUrl', async () => {
      const current = vscode.workspace
        .getConfiguration('misterpilot')
        .get<string>('backendUrl', 'http://localhost:8000');
      const url = await vscode.window.showInputBox({
        prompt: 'Enter the MisterPilot backend URL',
        placeHolder: 'http://localhost:8000',
        value: current,
        ignoreFocusOut: true,
        validateInput: (v) => {
          try { new URL(v); return null; } catch { return 'Enter a valid URL (e.g. http://localhost:8000)'; }
        },
      });
      if (url === undefined) return;
      await vscode.workspace
        .getConfiguration('misterpilot')
        .update('backendUrl', url, vscode.ConfigurationTarget.Global);
      vscode.window.showInformationMessage(`MisterPilot: Backend URL set to ${url}`);
    })
  );
}

export function deactivate(): void {}
