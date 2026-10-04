# Cómo publicar este repositorio (una sola vez, ~10 minutos)

> Hazlo **antes** de que aparezcan estaciones nuevas en Iztapalapa, Iztacalco o Tlalpan en el feed de Ecobici.
> La fecha que cuenta es la del servidor de GitHub (push y release), no la de tu computadora.

## 1. Crear el repositorio vacío
1. Entra a https://github.com/new
2. Nombre: `ecobici-red-cdmx` · **Public** · **no** marques "Add a README", ni .gitignore, ni licencia.
3. Crear.

## 2. Subir esta carpeta (PowerShell)
Requiere Git para Windows (https://git-scm.com/download/win). En PowerShell:

```powershell
cd "$HOME\OneDrive\Escritorio\ecobici_2026\ecobici-red-cdmx"
git init
git add .
git commit -m "Registro de predicciones de la expansión de Ecobici (4-oct-2026) y captura GBFS"
git branch -M main
git remote add origin https://github.com/TU_USUARIO/ecobici-red-cdmx.git
git push -u origin main
```
Cambia `TU_USUARIO`. Si pide credenciales, usa el inicio de sesión del navegador que abre Git.

Alternativa sin terminal: GitHub Desktop → "Add existing repository" → esta carpeta → "Publish repository"
(desmarca "Keep this code private").

## 3. Instalar la tarea de captura
Por seguridad no pude escribir directamente la carpeta `.github` en tu computadora, así que el archivo viene en
`github_workflow/captura_gbfs.yml`. Instálalo desde la web (es lo más simple):
1. En tu repositorio: **Add file → Create new file**.
2. Nombre del archivo: `.github/workflows/captura_gbfs.yml` (escríbelo así, con las diagonales).
3. Pega el contenido completo de `github_workflow/captura_gbfs.yml` → **Commit changes**.

## 3b. Dar permiso de escritura a la captura
Repositorio → **Settings → Actions → General → Workflow permissions** → "Read and write permissions" → Save.

## 4. Probar la captura
Pestaña **Actions** → "Captura estado Ecobici (GBFS)" → **Run workflow**. En ~1 minuto debe aparecer la rama
`gbfs-archive` con `status/AAAA/MM/DD/*.csv.gz` e `info/`. Después corre sola cada ~5 minutos.

## 5. Sellar el registro con un release
Repositorio → **Releases → Draft a new release** → tag `registro-2026-10-04` → título
"Registro de predicciones · expansión Ecobici 2026" → en la descripción pega la huella:
`46140e98aea343dce4ad729e7b517843dca48f7a35e7dadbf6aaa5fde217f45b` → Publish.

## Notas
- GitHub desactiva las tareas programadas de un repositorio público tras 60 días sin actividad. Si ves un
  aviso en Actions, vuelve a habilitarla con un clic.
- Cada captura pesa ~2–6 KB; ~1–2 MB por día en la rama `gbfs-archive`. `main` se mantiene ligera.
- Para el análisis basta clonar la rama de datos:
  `git clone --single-branch --branch gbfs-archive https://github.com/TU_USUARIO/ecobici-red-cdmx.git gbfs`
