; Inno Setup: gera o instalador do Windows a partir da pasta criada pelo PyInstaller.
; Uso:  ISCC.exe /DOrigem="<pasta dist\HumanoOuIA>" /DSaida="<pasta de saída>" instalador.iss
; Não exige administrador: instala para o usuário atual (em %LOCALAPPDATA%\Programs).

#ifndef Origem
  #define Origem "..\..\..\_build\dist\HumanoOuIA"
#endif
#ifndef Saida
  #define Saida "..\..\Instalador"
#endif
#define Versao "1.0.0"

[Setup]
AppId=HumanoOuIA-PIIV-FATEC-Jundiai
AppName=Humano ou IA? - Análise de autoria textual
AppVersion={#Versao}
AppVerName=Humano ou IA? {#Versao}
AppPublisher=FATEC Jundiaí - Projeto Integrador IV
DefaultDirName={autopf}\Humano ou IA
DefaultGroupName=Humano ou IA
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir={#Saida}
OutputBaseFilename=HumanoOuIA_Setup_{#Versao}
SetupIconFile=icone.ico
UninstallDisplayIcon={app}\HumanoOuIA.exe
Compression=lzma2/ultra64
SolidCompression=yes
LZMAUseSeparateProcess=yes
WizardStyle=modern
ShowLanguageDialog=no

[Languages]
Name: "pt"; MessagesFile: "compiler:Languages\BrazilianPortuguese.isl"

[Tasks]
Name: "desktopicon"; Description: "Criar atalho na Área de Trabalho"; GroupDescription: "Atalhos:"

[Files]
Source: "{#Origem}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\Humano ou IA"; Filename: "{app}\HumanoOuIA.exe"; WorkingDir: "{app}"
Name: "{group}\Desinstalar Humano ou IA"; Filename: "{uninstallexe}"
Name: "{autodesktop}\Humano ou IA"; Filename: "{app}\HumanoOuIA.exe"; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
Filename: "{app}\HumanoOuIA.exe"; Description: "Abrir o Humano ou IA agora"; Flags: nowait postinstall skipifsilent

[Messages]
pt.WelcomeLabel2=Este assistente vai instalar a plataforma de análise de autoria textual "Humano ou IA?" (Projeto Integrador IV, FATEC Jundiaí).%n%nTudo roda localmente neste computador: nenhum texto é enviado para a internet.%n%nO resultado da plataforma é uma estimativa experimental, não uma prova de autoria.
