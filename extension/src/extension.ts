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
}

export function deactivate(): void {}
