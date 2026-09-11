// DSPy docs "Ask AI" chat, powered by Mixedbread toast-1.
// Loaded on every page via mkdocs.yml extra_javascript. The widget streams
// answers from docs/api/chat.js, which holds the API keys and grounds answers
// in the dspy-docs + dspy-code Mixedbread stores.
//
// Layout mirrors the "Ask DSPy" modal the site shipped before (centered
// 800px dialog, gray footer, gray-tinted user turns) so the change is the
// engine, not the UI. Theme follows Material's data-md-color-scheme.
(function () {
  "use strict";

  // Backend: the Vercel Edge Function shipped with the site (docs/api/chat.js)
  // on the same origin. Override with window.TOAST_CHAT_ENDPOINT (set from
  // extra.toast_chat_endpoint in mkdocs.yml) to use an external proxy. Under
  // `mkdocs serve` the widget targets the local dev server, since mkdocs does
  // not run the function.
  var ENDPOINT =
    window.TOAST_CHAT_ENDPOINT ||
    (location.hostname === "localhost" || location.hostname === "127.0.0.1"
      ? "http://localhost:8787" // `node docs/api/dev.mjs`
      : "/api/chat");

  // Logos inlined as data URIs so the widget makes no external asset requests.
  var DSPY_LOGO = "data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4gPHN2ZyBpZD0iTGF5ZXJfMSIgZGF0YS1uYW1lPSJMYXllciAxIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZlcnNpb249IjEuMSIgdmlld0JveD0iMCAwIDIwMDAgMjAwMCI+IDxkZWZzPiA8c3R5bGU+IC5jbHMtMSB7IGZpbGw6ICNlZjQwMzY7IHN0cm9rZS13aWR0aDogMHB4OyB9IDwvc3R5bGU+IDwvZGVmcz4gPHBhdGggY2xhc3M9ImNscy0xIiBkPSJNMTI5OS4yOCwxOTMxLjA4Yy0zOS43MywwLTcyLjA4LS4yNi05Mi41OC0uODktMTExLjk0LTMuNDQtMzAxLjQ2LTMuNDgtNDMwLjgyLTEuNzMtOTMuNzgsMS4yNy0zNzIuOS04LjE2LTUzOS42Ny0xMy43OS01OC4xLTEuOTYtMTA4LjI3LTMuNjYtMTE1LjM5LTMuNjZoLTMuMzdjLTI0LjQzLS4wNy04MS4zOC4yMy04OC40NS03Ny40Ni0yLjAzLTIyLjI2LS43Ni0xNzQuNDgsMS42Ni00NDEuNTksMS41OS0xNzUuOSwzLjM5LTM3NS4yOCwzLjM5LTUwMy4xMywwLTI4MS41OSwzLjQ1LTYzOS4yNiw1LjI1LTcxOC4yOS41OS0yNS43OCw0LjI5LTU0Ljc2LDI3LjI5LTc0LjI0LDE4LjY1LTE1Ljc4LDM5LjkzLTE3LjIxLDYwLjUxLTE4LjU4LDguODgtLjU5LDE4LjA3LTEuMjEsMjguNjctMi44Niw0Mi40My02LjYzLDI1Mi4wNC02LjA4LDUyMS42Ny00LjY4LDY0LjY1LjM0LDEyNS43Mi42NSwxNjkuMzguNjVzMTE3LjEyLS4zMiwyMDEuODgtLjY5YzI3Ny4wMi0xLjIsNjU2LjQxLTIuODUsNzMzLjE5LDIuNTQsMTQuNjMsMS4wMywyOCwxLjY5LDQwLjkzLDIuMzQsMzkuNDcsMS45Nyw3MC42NSwzLjUzLDk1Ljc4LDE0LjIxLDI0Ljg5LDEwLjU3LDU0LjM1LDM0Ljc5LDUzLjM2LDkwLjExLTEuMTksNjYuNTUtLjc0LDI4NS43NS0uMzUsNDc5LjEzLjE4LDg5LjUxLjM1LDE3NC4yMS4zNSwyMzcuMzQsMCwxNDQuMS01LjI3LDU0NS40OS04LjQzLDc4NS4zMS0xLjE4LDkwLjI2LTIuMDQsMTU1LjQ3LTIuMDQsMTY0LjA3LDAsMTUuMiwwLDQ2LjgzLTI3LjA2LDY2LjA2LTE4LjUzLDEzLjE3LTQwLjksMTQuMjItNjguMjIsMTQuMjItMTEuMzUsMC01Mi42NS41NC0xMDkuOCwxLjI4LTEzMS45NywxLjcyLTMzMS40OCw0LjMyLTQ1Ny4xNSw0LjMyWk05NTYuOTIsMTgzNy41NmM5Mi4zMiwwLDE4NS45My44NiwyNTIuNTUsMi45MSw4My4zNiwyLjU2LDM4NC4xOC0xLjM2LDU0NS43OS0zLjQ3LDU3LjUtLjc1LDk5LjA0LTEuMjksMTEwLjk3LTEuMjksMi4wMywwLDMuODktLjAxLDUuNTktLjAzLjIzLTIyLjI4Ljk2LTc4LjE0LDEuOTgtMTU1Ljc1LDIuOTQtMjIzLjg5LDguNDItNjQwLjYyLDguNDItNzg0LjEzLDAtNjMuMDgtLjE3LTE0Ny43Mi0uMzUtMjM3LjE1LS4zOS0xOTMuODEtLjg0LTQxMy40Ny4zNi00ODAuOTIuMDUtMy0uMDktNS4xNC0uMjctNi41OC0xMC42NC0zLjg0LTM5Ljc5LTUuMjktNjMuNjEtNi40OC0xMi43NC0uNjQtMjcuMTktMS4zNi00Mi43My0yLjQ1LTczLjQ1LTUuMTYtNDY2LjQ2LTMuNDYtNzI2LjUyLTIuMzItODQuODYuMzctMTU4LjE2LjY5LTIwMi4yNy42OXMtMTA1LjA4LS4zMi0xNjkuODUtLjY1Yy0xODYuNjMtLjk3LTQ2OC42NS0yLjQ0LTUwNy4zNSwzLjYxLTE0LjUyLDIuMjctMjYuNzQsMy4wOS0zNi41NSwzLjc0LTEuMjQuMDgtMi41NC4xNy0zLjg1LjI2LS4wOCwxLjQ4LS4xNCwzLjE1LS4xOSw1LjAzaDBjLTEuNzksNzguNjktNS4yMyw0MzUuMjYtNS4yMyw3MTYuMjYsMCwxMjguMjUtMS44LDMyNy44NS0zLjQsNTAzLjk0LTEuNzQsMTkyLjg3LTMuNTQsMzkxLjk5LTIuMjQsNDI4LjQ5aDIuNjNjOC4yNywwLDQ0LjI3LDEuMiwxMTguNDIsMy43LDE1NS4wOSw1LjI0LDQ0My43OSwxNS4wNCw1MzUuNDMsMTMuNzUsNTMuNjctLjczLDExNy42NS0xLjE0LDE4Mi4yNS0xLjE0WiIvPiA8cGF0aCBjbGFzcz0iY2xzLTEiIGQ9Ik01MTcuODksMTI1OC4zNWMtNTAuODMsMC04Ny40OC02LjkyLTExNS41Mi0yMS40MS00MS4zNS0yMS4zNy02My43Ni01OC4zMi02OC41MS0xMTIuOTctMy4xOS0zNi43LDEuNS02OS4wMiw4LjIxLTk0LjUtMjMuMDUtMi44Ni01MC4xMy01LjcxLTc0LjY4LTctNTguMS0zLjA3LTE4NS43NSwxLjU5LTE4Ny4wMiwxLjYzbC0zLjMxLTg5LjdjNS40LS4yLDEzMi45NS00LjgzLDE5NS4wNS0xLjU3LDYyLjU1LDMuMywxMzYuNDIsMTUuMjksMTM5LjUzLDE1LjgsMTQuODIsMi40MiwyNy42MiwxMS43OSwzMy44MiwyNS40Niw2LjIsMTMuNjcsNS4zMywyOS4yNC0yLjYxLDQxLjk4LS4wNy4xMy0yNC41Myw0My4wMy0xOS41NiwxMDAuMTIsMi44LDMyLjIxLDUuMjIsNjAuMDYsMTM5LjM5LDUwLjY3LDM2LjQtMi41NCw0Ni44LTE2LjI5LDUwLjIyLTIwLjgxLDIyLjU4LTI5Ljg3LDkuNC05Mi43MywzLjc4LTExOS41N2wtLjkzLTQuNDVjLTYuMTgtMjkuODQtMS4xMi01My42NSwxNS4wMy03MC43NCwyNS4xMS0yNi41OCw2MS4yNS0yMS4yMiw4Ny42Mi0xNy4zMmw1LjguODVjNy40NCwxLjA2LDI1LjM2LDMuODksNTIuNDksOC4xNyw5NS4xMSwxNC45OSwyOTMuMDIsNDYuMTksMzY1Ljg1LDUyLjM2LDQ3LjIzLDQsOTUuMjIsMy44MSwxMzAuMTgsMi43MS0xLjU1LTguMDgtMy0xNi42Ny00LjIxLTI1LjU0LTkuMjUtNjcuODEuMDMtMTE1LjA2LDI4LjM2LTE0NC40NCw1Ny4yNy01OS4zOSwyMDkuNjQtODUuMTIsMjk5LjI3LTI5LjM2LDY2LjM5LDQxLjMxLDU4LDEyNi45Myw0MS4yLDE4Ny43NCwxMTUuNjQtNi43OCwyODIuNTUtMjYuMDgsMjg0LjU5LTI2LjMybDEwLjM2LDg5LjE2Yy0xMC4yOCwxLjItMjUyLjg3LDI5LjI1LTM1OS44NiwyOS4yNS0xNS4zOSwwLTI5LjctNy44OC0zNy45My0yMC44OC04LjIzLTEzLTkuMjItMjkuMzEtMi42My00My4yMiwyMi43Ni00OC40Myw0MC4yMS0xMjQuOTgsMTYuODUtMTM5LjUyLTIzLjk2LTE0LjkxLTYyLjU4LTIwLjg3LTEwMy4zNC0xNS45NS00MS42OCw1LjAzLTcyLjcxLDE5LjgxLTgzLjg5LDMxLjQtMy4yNywzLjM5LTEwLjM4LDIyLjM3LTQuMTksNjguOTIsNC43MiwzNS40NCwxNC41Nyw2Ni45MSwxNC42Nyw2Ny4yMiw0LjEsMTMuMDEsMi4wOCwyNy4yMS01LjUzLDM4LjU0LTcuNiwxMS4zMy0xOS45NCwxOC42MS0zMy41MywxOS43NC00LjA3LjM0LTEwMC44OSw4LjIzLTE5Ny45NiwwLTc2LjA0LTYuNDQtMjc2LjEtMzcuOTgtMzcyLjI1LTUzLjEzLTI1Ljc4LTQuMDYtNDQuNDEtNy01MS4yLTcuOTdsLTMuOS0uNTdjMy41LDE4LjAxLDcuMzMsNDEuODIsNy43NSw2Ny4wNi43NCw0NC45NC05LjYzLDgxLjk1LTMwLjg0LDExMC0yNS41MSwzMy43NC02NC4zOSw1Mi42Ny0xMTUuNTgsNTYuMjQtMTguNDYsMS4yOS0zNS40MiwxLjk0LTUxLjA0LDEuOTRaTTcwMi45NCw5OTguODFoMHMwLDAsMCwwWiIvPiA8cGF0aCBjbGFzcz0iY2xzLTEiIGQ9Ik0xMDE3LjgyLDE4ODYuNTJsLTg5Ljc3LS42NmMuMDItMi4yNCwxLjYyLTIyNC45OC4wMy0yNjcuOTItLjUyLTEzLjg4LTIuMDgtNTYuMTIsMzAuODEtNzguNjksMzEuODgtMjEuODksNzIuMzktOS4wOCwxMDQuMTEsNC41MSwzMy42MSwxNC40MSw2Ny45NiwxNy4yNSw4Ny41Miw3LjI1LDcuNTQtMy44NSwxOC4xOS0xMS44NSwyMy43Ny0zNi42OSwxMy40LTU5Ljc4LDEuOTMtMTE1Ljk5LTE2LjMyLTE0MC42Mi04LjQ5LTExLjQ2LTE1LjExLTExLjk1LTE3LjI4LTEyLjEyLTQ1LjQ2LTMuNC0xMTAuNjUsMTYuOTgtMTMxLjk1LDI1LjEyLTIwLjgsNy45Ni00NC4zNC0uNTMtNTUuMjItMTkuOTYtNS4yLTkuMjYtMzAuODMtNjIuMjctMjUuNTUtMTk2LjQ3LDMuMzktODYuMzUsMjYuNzQtMjMzLjIsNDMuNzgtMzQwLjQyLDYuOTMtNDMuNTcsMTQuMDctODguNTEsMTUuMzgtMTAyLjk5LTQzLjA4LDYuNjYtMTAxLjUzLDIuMjktMTQ0Ljk4LTMwLjE1LTI1LjYxLTE5LjEyLTU2Ljg5LTU2LjI0LTYwLjQ4LTEyNC41MS0zLjUyLTY2LjksMTAuNDYtMTIwLjA5LDQxLjU1LTE1OC4wOSwzMi45Ny00MC4zLDc0LjctNTAuNzMsOTYuNjQtNTMuMzcsMjUuNzUtMy4xLDUwLjY2LS4yMyw3MC41NSwyLjczLDQuNjktNTIuMDYsNy44Ny0xNTcuNjUsOC4zOC0yNDguNjlsODkuNzcuNWMwLC43NC0uNDQsNzQuODctMi45MSwxNDguNDUtMS40Niw0My42Ni0zLjM2LDc4Ljc3LTUuNjIsMTA0LjM0LTMuNDksMzkuNC03LjEsODAuMTQtNDUuNzYsODcuODMtMTQuNzIsMi45Mi0zMC4xOC41Mi00OC4xLTIuMjctMTcuNzMtMi43Ni0zNy44Mi01LjktNTUuNjEtMy43Ni00MS4zMiw0Ljk2LTYyLjkyLDQ3LjgzLTU5LjI0LDExNy42MSwxLjQzLDI3LjI5LDkuNDYsNDYuMDQsMjQuNTUsNTcuMjksMjQuMDgsMTcuOTgsNjQuODIsMTYuODIsODYuMzQsMTEuNjYsMTIuNDktMi45OSw0NS42Ny0xMC45Niw3MS41NCwxMC4zNywyNS44NywyMS4zMiwyNC40OCw1My4xOSwyMy4yNiw4MS4zLS42NSwxNS4wMy02LjE0LDUwLjA2LTE2LjYsMTE1Ljg0LTE1Ljc1LDk5LjA4LTM5LjU1LDI0OC44Mi00Mi43NCwzMjkuODYtMi4wOSw1My4yMiwxLjMyLDkwLjE2LDUuNCwxMTQuMSwzNC4xNC05LjM5LDgxLjU2LTE5LjAyLDEyNC4zMy0xNS44MywzMi41NSwyLjQ0LDYxLjE0LDE5LjEsODIuNjksNDguMTksMzcuMTcsNTAuMTYsNDkuNjUsMTM0LjA0LDMxLjc5LDIxMy43LTEyLjc1LDU2Ljg4LTQ1LjM3LDg0LjEzLTcwLjQ5LDk2Ljk4LTY0LjY3LDMzLjA3LTE0MS42OSw0Ljc5LTE2My43NS00LjY3LTMuNjEtMS41NS02LjgyLTIuODMtOS42NC0zLjksMS4zLDU3Ljg5LS4xMywyNTUuMzUtLjE5LDI2NC4xNVoiLz4gPC9zdmc+";
  var MXB_LOGO = "data:image/png;base64,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";

  var history = [];

  var host = document.createElement("div");
  host.id = "toast-chat";
  var root = host.attachShadow({ mode: "open" });
  root.innerHTML =
    "<style>" +
    ":host{all:initial}" +
    "*{box-sizing:border-box;margin:0;font-family:Inter,ui-sans-serif,system-ui,-apple-system,'Segoe UI',sans-serif}" +
    ".wrap{--fg:#222222;--bg:#ffffff;--muted:#6b6b6b;--line:#e5e5e5;--soft:#f4f4f4;" +
    "--user:rgba(238,238,238,.4);--code:rgba(10,64,255,.08);--overlay:rgba(0,0,0,.5);" +
    "--primary:#111111;--primary-fg:#ffffff;--send-off:#bdbdbd;font-size:14px;color:var(--fg)}" +
    ".wrap.dark{--fg:#ededf0;--bg:#16171b;--muted:#9a9aa3;--line:#2e2f36;--soft:#1f2026;" +
    "--user:rgba(255,255,255,.05);--code:rgba(120,160,255,.14);--overlay:rgba(0,0,0,.65);" +
    "--primary:#f2f2f2;--primary-fg:#111111;--send-off:#4a4b52}" +

    // floating trigger (bottom right)
    ".fab{position:fixed;bottom:24px;right:24px;z-index:2147483000;display:flex;align-items:center;gap:8px;" +
    "cursor:pointer;background:var(--bg);color:var(--fg);border:1.5px solid var(--fg);" +
    "border-radius:999px;padding:10px 20px 10px 16px;font-size:18px;font-weight:400;line-height:24px;" +
    "box-shadow:0 4px 14px rgba(0,0,0,.18);transition:transform .15s ease,box-shadow .15s ease}" +
    ".fab:hover{transform:translateY(-1px);box-shadow:0 6px 18px rgba(0,0,0,.24)}" +
    ".fab img{width:22px;height:22px;display:block}" +
    ".wrap.open .fab{opacity:0;pointer-events:none}" +

    // overlay + centered dialog
    ".overlay{position:fixed;inset:0;z-index:2147483001;background:var(--overlay);" +
    "opacity:0;pointer-events:none;transition:opacity .18s ease}" +
    ".wrap.open .overlay{opacity:1;pointer-events:auto}" +
    ".panel{position:fixed;left:50%;top:50%;transform:translate(-50%,-50%) scale(.98);z-index:2147483002;" +
    "width:800px;max-width:calc(100vw - 32px);max-height:90vh;display:flex;flex-direction:column;" +
    "background:var(--bg);color:var(--fg);border-radius:16px;" +
    "box-shadow:0 10px 40px rgba(0,0,0,.22);overflow:hidden;opacity:0;pointer-events:none;" +
    "transition:opacity .18s ease,transform .18s ease}" +
    ".wrap.open .panel{opacity:1;pointer-events:auto;transform:translate(-50%,-50%) scale(1)}" +
    ".wrap.wide .panel{width:1100px;max-height:96vh}" +

    // header
    ".head{display:flex;align-items:center;gap:6px;padding:25px 24px 21px}" +
    ".head .mark{width:24px;height:24px;margin-right:6px;display:flex;align-items:center;justify-content:center}" +
    ".head .mark img{width:100%;height:100%;display:block}" +
    ".head h2{font-size:24px;font-weight:600;line-height:32px;flex:1;min-width:0;white-space:nowrap;" +
    "overflow:hidden;text-overflow:ellipsis}" +
    ".hbtn{width:40px;height:40px;border:none;background:transparent;color:var(--fg);cursor:pointer;" +
    "border-radius:999px;display:flex;align-items:center;justify-content:center;flex:none;" +
    "transition:background .15s ease,transform .2s ease}" +
    ".hbtn:hover{background:var(--soft)}" +
    ".hbtn svg{width:18px;height:18px}" +
    ".wrap.wide .hbtn.expand{transform:rotate(180deg)}" +

    // messages (hidden until the first question, like the original modal)
    ".msgs{display:none;flex:1;min-height:0;overflow-y:auto;padding:0 16px 12px;flex-direction:column;gap:12px;" +
    "scrollbar-width:thin;scrollbar-color:var(--line) transparent}" +
    ".wrap.has-msgs .msgs{display:flex}" +
    ".msgs::-webkit-scrollbar{width:6px}" +
    ".msgs::-webkit-scrollbar-thumb{background:var(--line);border-radius:3px}" +
    ".turn{width:100%;padding:18px 16px;border-radius:12px;animation:rise .2s ease}" +
    ".turn.bot{padding-top:8px}" +
    "@keyframes rise{from{opacity:0;transform:translateY(4px)}to{opacity:1;transform:none}}" +
    ".turn.user{background:var(--user)}" +
    ".turn.bot{background:var(--bg)}" +
    ".who{display:flex;align-items:center;gap:8px;font-size:13px;font-weight:700;margin-bottom:8px}" +
    ".who img,.who svg{width:16px;height:16px;display:block}" +
    ".body{padding-left:24px;font-size:15px;line-height:1.65;overflow-wrap:break-word}" +
    ".turn.user .body{white-space:pre-wrap}" +
    ".body p{margin:0 0 14px}.body p:last-child,.body ul:last-child,.body ol:last-child{margin-bottom:0}" +
    ".body ul,.body ol{margin:0 0 14px;padding-left:22px}.body li{margin:6px 0}" +
    ".body li>ul,.body li>ol{margin:6px 0 0}" +
    ".body h3{font-size:16px;font-weight:650;margin:18px 0 8px}" +
    ".body h4{font-size:15px;font-weight:600;margin:16px 0 6px}" +
    ".body h3:first-child,.body h4:first-child{margin-top:0}" +
    ".body hr{border:none;border-top:1px solid var(--line);margin:10px 0}" +
    ".body .tblwrap{overflow-x:auto;max-width:100%;margin:8px 0}" +
    ".body table{border-collapse:collapse;font-size:13px;line-height:1.45}" +
    ".body th,.body td{border:1px solid var(--line);padding:4px 8px;text-align:left;vertical-align:top}" +
    ".body th{background:var(--soft);font-weight:600}" +
    ".body code{background:var(--code);padding:2px 6px;border-radius:6px;" +
    "font-size:13px;font-family:'Fira Mono',ui-monospace,SFMono-Regular,Menlo,monospace}" +
    ".cb{background:#0d1126;border-radius:12px;margin:16px 0;overflow:hidden;max-width:100%}" +
    ".cbh{display:flex;align-items:center;justify-content:space-between;height:40px;padding:0 10px 0 16px;" +
    "background:rgba(255,255,255,.035);border-bottom:1px solid rgba(255,255,255,.08)}" +
    ".cbh .lang{font-size:12px;color:#8b93a7;letter-spacing:.01em}" +
    ".copy{width:30px;height:30px;border:none;background:transparent;color:#8b93a7;cursor:pointer;border-radius:6px;" +
    "display:flex;align-items:center;justify-content:center;transition:background .15s ease,color .15s ease}" +
    ".copy:hover{background:rgba(255,255,255,.08);color:#e6e9f2}.copy svg{width:15px;height:15px}" +
    ".copy.ok{color:#5fd38d}" +
    ".body pre{background:transparent;color:#e6e9f2;padding:16px 18px;margin:0;overflow-x:auto;" +
    "font-size:14px;line-height:1.7;scrollbar-width:thin;scrollbar-color:rgba(255,255,255,.25) transparent}" +
    ".body pre::-webkit-scrollbar{height:6px}.body pre::-webkit-scrollbar-thumb{background:rgba(255,255,255,.22);border-radius:3px}" +
    ".body pre code{background:none;border:none;padding:0;color:inherit;font-size:inherit;" +
    "font-family:'Fira Mono',ui-monospace,SFMono-Regular,Menlo,monospace}" +
    // the * reset above sets Inter on every element; highlighted spans must keep the mono face
    ".body pre code *,.body code *{font-family:inherit;font-size:inherit}" +
    ".c-kw{color:#5fd38d}.c-ty{color:#79c0ff}.c-fn{color:#79c0ff}.c-str{color:#a5e2a0}.c-con{color:#ff7b72}" +
    ".c-com{color:#8b949e;font-style:italic}.c-num{color:#ffa657}.c-dec{color:#d2a8ff}" +
    ".body a{color:var(--fg);text-decoration:underline;text-underline-offset:2px}" +
    ".body .cut{margin-top:12px;font-size:13px;color:var(--muted);font-style:italic}" +

    // status line while searching
    ".status{display:flex;align-items:center;gap:8px;font-size:13px;color:var(--muted);padding:0 16px 8px 40px}" +
    ".status .stxt{max-width:520px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;font-style:italic}" +
    ".dots{display:inline-flex;gap:3px}" +
    ".dots i{width:4px;height:4px;border-radius:50%;background:var(--fg);opacity:.4;animation:blink 1.2s infinite}" +
    ".dots i:nth-child(2){animation-delay:.2s}.dots i:nth-child(3){animation-delay:.4s}" +
    "@keyframes blink{0%,80%,100%{opacity:.35}40%{opacity:1}}" +

    // trace block styled like the original "Sources" block
    ".trace{margin:16px 0 4px;background:var(--soft);border-radius:12px;padding:14px 16px}" +
    ".trace summary{cursor:pointer;font-size:14px;font-weight:700;user-select:none;list-style:none;" +
    "display:flex;align-items:center;justify-content:space-between}" +
    ".trace summary::after{content:'';width:8px;height:8px;border-right:2px solid var(--muted);" +
    "border-bottom:2px solid var(--muted);transform:rotate(45deg);transition:transform .15s ease;margin-right:4px}" +
    ".trace[open] summary::after{transform:rotate(-135deg)}" +
    ".trace-body{margin-top:10px;font-size:13px;color:var(--muted);line-height:1.55;white-space:pre-wrap}" +
    ".trace-q{margin-top:8px;display:flex;gap:8px;align-items:baseline;font-size:13px;color:var(--fg);" +
    "font-family:'Fira Mono',ui-monospace,SFMono-Regular,Menlo,monospace}" +
    ".trace-q::before{content:'\\2315';font-size:13px;color:var(--muted)}" +

    // input
    ".foot{padding:0 16px 14px}" +
    ".wrap.has-msgs .foot{padding-top:6px}" +
    ".box{position:relative;display:flex;align-items:center;gap:12px;min-height:56px;" +
    "border:2px solid var(--fg);border-radius:8px;background:var(--bg);padding:10px 14px 10px 16px;" +
    "transition:border-color .15s ease}" +
    ".clip{width:20px;height:20px;flex:none;color:var(--muted);display:flex;align-items:center;justify-content:center}" +
    ".clip svg{width:18px;height:18px}" +
    "textarea{flex:1;resize:none;border:none;background:transparent;color:var(--fg);outline:none;" +
    "font-size:16px;line-height:24px;height:24px;max-height:144px;padding:0}" +
    "textarea::placeholder{color:var(--muted)}" +
    ".send{width:32px;height:32px;flex:none;border:none;border-radius:50%;cursor:pointer;" +
    "background:var(--primary);color:#fff;display:flex;align-items:center;justify-content:center;" +
    "transition:background .15s ease,transform .15s ease}" +
    ".send:hover:not(:disabled){transform:scale(1.04)}" +
    ".send:disabled{background:var(--send-off);cursor:default}" +
    ".wrap.dark .send:not(:disabled){color:var(--primary-fg)}" +
    ".send svg{width:14px;height:14px;display:block;margin-left:2px}" +

    // footer bar
    ".brand{height:56px;padding:0 24px;background:var(--soft);display:flex;align-items:center;" +
    "justify-content:space-between;gap:12px;font-size:10px;color:var(--muted)}" +
    ".brand .left{display:flex;align-items:center;gap:6px;min-width:0}" +
    ".brand a{display:inline-flex;align-items:center;gap:6px;color:var(--muted);text-decoration:none}" +
    ".brand a:hover{color:var(--fg)}" +
    ".brand img{height:16px;width:auto;display:block}" +
    ".brand .right{display:flex;align-items:center;gap:12px;white-space:nowrap}" +
    ".brand .round{width:32px;height:32px;border-radius:50%;background:var(--bg);border:1px solid var(--line);" +
    "display:flex;align-items:center;justify-content:center;color:#5865F2}" +
    ".brand .round svg{width:16px;height:16px}" +
    ".tip{position:absolute;right:24px;top:-30px;background:var(--fg);color:var(--bg);font-size:11px;" +
    "padding:4px 8px;border-radius:6px;opacity:0;transition:opacity .15s ease;pointer-events:none}" +
    ".head{position:relative}.tip.show{opacity:1}" +
    ".brand kbd{font-family:'Fira Mono',ui-monospace,monospace;font-size:10px;border:1px solid var(--line);" +
    "border-radius:4px;padding:1px 5px;background:var(--bg);color:var(--muted)}" +

    "@media (max-width:640px){.panel{max-height:100vh;border-radius:12px}.head h2{font-size:20px}}" +
    "@media (prefers-reduced-motion: reduce){*{animation:none!important;transition:none!important}}" +
    "</style>" +
    '<div class="wrap">' +
    '<button class="fab" aria-label="Ask AI"><img src="' + DSPY_LOGO + '" alt=""> Ask AI</button>' +
    '<div class="overlay"></div>' +
    '<div class="panel" role="dialog" aria-label="Ask DSPy">' +
    '<div class="head"><div class="mark"><img src="' + DSPY_LOGO + '" alt="DSPy"></div>' +
    "<h2>Ask DSPy</h2>" +
    '<span class="tip">Link copied</span>' +
    '<button class="hbtn expand" title="Expand" aria-label="Expand">' +
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M15 6l-6 6 6 6"/></svg></button>' +
    '<button class="hbtn share" title="Copy link to this chat" aria-label="Copy link">' +
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">' +
    '<path d="M10 13a5 5 0 0 0 7.5.5l3-3a5 5 0 0 0-7-7l-1.5 1.5"/><path d="M14 11a5 5 0 0 0-7.5-.5l-3 3a5 5 0 0 0 7 7L12 19"/></svg></button>' +
    '<button class="hbtn new" title="Clear chat" aria-label="Clear chat">' +
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">' +
    '<path d="M3 6h18M8 6V4a1 1 0 0 1 1-1h6a1 1 0 0 1 1 1v2m2 0v14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2V6h12zM10 11v6M14 11v6"/></svg></button>' +
    "</div>" +
    '<div class="msgs"></div>' +
    '<div class="foot"><div class="box">' +
    '<span class="clip" title="Attachments are not supported">' +
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">' +
    '<path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48"/></svg></span>' +
    '<textarea rows="1" placeholder="Ask DSPy a question..." aria-label="Ask DSPy a question"></textarea>' +
    '<button class="send" title="Send" aria-label="Send" disabled>' +
    '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z"/></svg></button>' +
    "</div></div>" +
    '<div class="brand"><div class="left"><span>Powered by</span>' +
    '<a href="https://www.mixedbread.com" target="_blank" rel="noopener"><img src="' + MXB_LOGO + '" alt="Mixedbread"></a></div>' +
    '<div class="right"><a href="https://www.mixedbread.com/blog/toast-1" target="_blank" rel="noopener">toast-1</a>' +
    '<a class="round" href="https://discord.gg/XCGy2WDCQB" target="_blank" rel="noopener" title="DSPy Discord">' +
    '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M20.3 4.4A19.8 19.8 0 0 0 15.4 3l-.2.4a18 18 0 0 1 4.5 2.3 15 15 0 0 0-15.4 0A18 18 0 0 1 8.8 3.4L8.6 3a19.8 19.8 0 0 0-4.9 1.5C.6 9.1-.2 13.6.2 18.1a19.9 19.9 0 0 0 6 3l1.3-1.8a12.9 12.9 0 0 1-2-1l.5-.4a14.2 14.2 0 0 0 12 0l.5.4a12.9 12.9 0 0 1-2 1l1.3 1.8a19.9 19.9 0 0 0 6-3c.5-5.2-.8-9.7-3.5-13.7zM8.5 15.4c-1.2 0-2.1-1.1-2.1-2.4s.9-2.4 2.1-2.4 2.1 1.1 2.1 2.4-.9 2.4-2.1 2.4zm7 0c-1.2 0-2.1-1.1-2.1-2.4s.9-2.4 2.1-2.4 2.1 1.1 2.1 2.4-.9 2.4-2.1 2.4z"/></svg></a>' +
    "</div></div>" +
    "</div></div>";

  var wrap = root.querySelector(".wrap");
  var fab = root.querySelector(".fab");
  var overlay = root.querySelector(".overlay");
  var msgs = root.querySelector(".msgs");
  var input = root.querySelector("textarea");
  var send = root.querySelector(".send");

  function newChat() {
    if (wrap.classList.contains("busy")) return; // don't clear mid-answer
    history = [];
    msgs.innerHTML = "";
    wrap.classList.remove("has-msgs");
    resetInput();
    input.focus();
  }

  // Follow Material's palette (data-md-color-scheme on <body>); fall back to
  // the OS preference when the page hasn't declared one.
  function theme() {
    var scheme = document.body && document.body.getAttribute("data-md-color-scheme");
    var dark = scheme === "slate"
      ? true
      : scheme === "default"
        ? false
        : window.matchMedia && matchMedia("(prefers-color-scheme: dark)").matches;
    wrap.classList.toggle("dark", dark);
  }
  function watchTheme() {
    new MutationObserver(theme).observe(document.body, {
      attributes: true,
      attributeFilter: ["data-md-color-scheme"],
    });
    theme();
  }

  // textarea grows with input (up to max-height); send enables when ready
  function syncInput() {
    input.style.height = "22px";
    input.style.height = Math.min(input.scrollHeight, 132) + "px";
    send.disabled = !input.value.trim() || wrap.classList.contains("busy");
  }
  function resetInput() {
    input.value = "";
    input.style.height = "22px";
    send.disabled = true;
  }
  input.addEventListener("input", syncInput);

  function setOpen(open) {
    wrap.classList.toggle("open", open);
    if (open) setTimeout(function () { input.focus(); }, 30);
  }
  fab.addEventListener("click", function () { setOpen(true); });
  overlay.addEventListener("click", function () { setOpen(false); });
  root.querySelector(".hbtn.new").addEventListener("click", newChat);
  root.querySelector(".hbtn.expand").addEventListener("click", function () {
    wrap.classList.toggle("wide");
  });
  // share: a link that reopens this page with the first question asked
  root.querySelector(".hbtn.share").addEventListener("click", function () {
    var first = history.length ? history[0].content : input.value.trim();
    var url = location.href.split("#")[0] + (first ? "#ask=" + encodeURIComponent(first) : "");
    var tip = root.querySelector(".tip");
    function done() {
      tip.classList.add("show");
      setTimeout(function () { tip.classList.remove("show"); }, 1400);
    }
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(url).then(done, done);
    } else {
      done();
    }
  });
  document.addEventListener("keydown", function (e) {
    if (e.key === "Escape") setOpen(false);
    // Mod+J toggles the dialog, the shortcut the previous docs widget used
    if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "j") {
      e.preventDefault();
      setOpen(!wrap.classList.contains("open"));
    }
  });

  function esc(s) {
    return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }

  // tiny Python-leaning highlighter (no external libs); runs on escaped
  // text, so entities like &lt; pass through untouched
  var HL_RE = new RegExp(
    "(#[^\\n]*)" + // comment
      "|(\"\"\"[\\s\\S]*?\"\"\"|\"(?:[^\"\\\\\\n]|\\\\.)*\"|'(?:[^'\\\\\\n]|\\\\.)*')" + // string
      "|\\b(def|class|return|import|from|as|if|elif|else|for|while|in|not|and|or|" +
      "with|try|except|finally|raise|lambda|yield|async|await|pass|break|continue|" +
      "global|nonlocal|is|del|assert)\\b" + // keyword
      "|\\b(None|True|False|self)\\b" + // constant
      "|\\b([A-Z][A-Za-z0-9_]*)\\b" + // class / type
      "|\\b([a-z_][A-Za-z0-9_]*)(?=\\()" + // function call / definition name
      "|\\b(\\d[\\d_]*(?:\\.\\d+)?)\\b" + // number
      "|(@[A-Za-z_][\\w.]*)", // decorator
    "g"
  );

  function hl(src) {
    return esc(src).replace(HL_RE, function (m, com, str, kw, con, ty, fn, num, dec) {
      var cls = com ? "c-com" : str ? "c-str" : kw ? "c-kw" : con ? "c-con" : ty ? "c-ty"
        : fn ? "c-fn" : num ? "c-num" : "c-dec";
      return '<span class="' + cls + '">' + m + "</span>";
    });
  }

  var COPY_ICON =
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">' +
    '<rect x="9" y="9" width="12" height="12" rx="2"/><path d="M5 15V5a2 2 0 0 1 2-2h10"/></svg>';
  var CHECK_ICON =
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12l5 5L20 7"/></svg>';

  function codeBlock(lang, lines) {
    return '<div class="cb"><div class="cbh"><span class="lang">' + esc(lang || "code") + "</span>" +
      '<button class="copy" title="Copy code" aria-label="Copy code">' + COPY_ICON + "</button></div>" +
      "<pre><code>" + hl(lines.join("\n")) + "</code></pre></div>";
  }

  // minimal markdown, parsed line by line so partial streamed input can't
  // derail it: fences open only at line starts, ordered lists keep the
  // author's numbering via <li value>, plus nested lists, tables, headings,
  // hr, inline code / bold / italic / links
  function inline(s) {
    return esc(s)
      .replace(/`([^`\n]+)`/g, "<code>$1</code>")
      .replace(/\*\*([^*]+)\*\*/g, "<b>$1</b>")
      .replace(/(^|[\s(])\*([^*\n]+)\*(?=$|[\s).,;:!?])/g, "$1<i>$2</i>")
      .replace(/\[([^\]]+)\]\((https?:[^)\s]+)\)/g, '<a href="$2" target="_blank">$1</a>');
  }

  function md(s) {
    var html = "";
    var para = []; // pending paragraph lines (already inlined)
    var tbl = null; // pending table lines (raw)
    var code = null; // pending fence body lines while inside ```
    var codeLang = ""; // language from the opening fence
    var codeIndent = ""; // indentation of the opening fence (fences inside lists)
    var lists = []; // open lists: {tag, indent, liOpen}

    function flushPara() {
      if (para.length) {
        html += "<p>" + para.join("<br>") + "</p>";
        para = [];
      }
    }
    function cells(row) {
      return row
        .replace(/^\s*\|/, "")
        .replace(/\|\s*$/, "")
        .split("|")
        .map(function (c) { return c.trim(); });
    }
    function flushTable() {
      if (!tbl) return;
      var rows = tbl;
      tbl = null;
      if (rows.length >= 2 && /^[\s:|-]+$/.test(rows[1]) && rows[1].indexOf("-") !== -1) {
        html += '<div class="tblwrap"><table><tr>' +
          cells(rows[0]).map(function (c) { return "<th>" + inline(c) + "</th>"; }).join("") +
          "</tr>" +
          rows.slice(2).map(function (r) {
            return "<tr>" +
              cells(r).map(function (c) { return "<td>" + inline(c) + "</td>"; }).join("") +
              "</tr>";
          }).join("") +
          "</table></div>";
      } else {
        // no separator row (yet): fall back to plain lines
        para = para.concat(rows.map(inline));
        flushPara();
      }
    }
    function closeList() {
      var l = lists.pop();
      html += (l.liOpen ? "</li>" : "") + "</" + l.tag + ">";
    }
    function closeAllLists() {
      while (lists.length) closeList();
    }

    var lines = s.split("\n");
    for (var i = 0; i < lines.length; i++) {
      var line = lines[i];

      if (code) {
        if (/^\s*```/.test(line)) {
          html += codeBlock(codeLang, code);
          code = null;
        } else {
          // drop the fence's own indentation so list-nested code is flush
          code.push(codeIndent && line.indexOf(codeIndent) === 0 ? line.slice(codeIndent.length) : line);
        }
        continue;
      }
      var fence = /^(\s*)```\s*([\w.+-]*)/.exec(line);
      if (fence) {
        flushPara();
        flushTable();
        closeAllLists();
        code = [];
        codeIndent = fence[1];
        codeLang = fence[2].toLowerCase();
        continue;
      }

      // GFM tables: a row starts with a pipe; the trailing pipe is optional
      // (and absent while a row is still streaming)
      if (/^\s*\|/.test(line)) {
        flushPara();
        (tbl = tbl || []).push(line);
        continue;
      }
      flushTable();

      if (!line.trim()) {
        flushPara(); // blank lines end paragraphs but not lists
        continue;
      }

      var li = /^(\s*)([-*]|\d+[.)])\s+(.*)$/.exec(line);
      if (li) {
        flushPara();
        var indent = li[1].replace(/\t/g, "    ").length;
        var tag = /\d/.test(li[2]) ? "ol" : "ul";
        var top = lists[lists.length - 1];
        while (top && indent < top.indent - 1) {
          closeList();
          top = lists[lists.length - 1];
        }
        if (top && indent <= top.indent + 1) {
          if (top.liOpen) {
            html += "</li>";
            top.liOpen = false;
          }
          if (top.tag !== tag) {
            closeList();
            top = lists[lists.length - 1];
          }
        }
        top = lists[lists.length - 1];
        if (!top || indent > top.indent + 1) {
          html += "<" + tag + ">"; // nested lists live inside the open <li>
          lists.push({ tag: tag, indent: indent, liOpen: false });
          top = lists[lists.length - 1];
        }
        html += "<li" +
          (tag === "ol" ? ' value="' + parseInt(li[2], 10) + '"' : "") +
          ">" + inline(li[3]);
        top.liOpen = true;
        continue;
      }

      var h = /^\s*(#{1,6})\s+(.*)$/.exec(line);
      if (h) {
        flushPara();
        closeAllLists();
        var hTag = h[1].length <= 2 ? "h3" : "h4";
        html += "<" + hTag + ">" + inline(h[2]) + "</" + hTag + ">";
        continue;
      }
      if (/^\s*([-*_])\s*(\1\s*){2,}$/.test(line)) {
        flushPara();
        closeAllLists();
        html += "<hr>";
        continue;
      }

      closeAllLists();
      para.push(inline(line));
    }

    if (code) html += codeBlock(codeLang, code); // fence still streaming
    flushPara();
    flushTable();
    closeAllLists();
    return html;
  }

  var USER_ICON =
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">' +
    '<circle cx="12" cy="8" r="4"/><path d="M4 21a8 8 0 0 1 16 0"/></svg>';

  function turn(cls, text) {
    var el = document.createElement("div");
    el.className = "turn " + cls;
    var who = document.createElement("div");
    who.className = "who";
    who.innerHTML = cls === "user" ? USER_ICON + "You" : '<img src="' + DSPY_LOGO + '" alt="">DSPy';
    var body = document.createElement("div");
    body.className = "body";
    if (cls === "user") body.textContent = text;
    else body.innerHTML = md(text);
    el.appendChild(who);
    el.appendChild(body);
    msgs.appendChild(el);
    wrap.classList.add("has-msgs");
    msgs.scrollTop = msgs.scrollHeight;
    return body;
  }

  function askToast() {
    var q = input.value.trim();
    if (!q || wrap.classList.contains("busy")) return;
    resetInput();
    wrap.classList.add("busy");
    turn("user", q);
    history.push({ role: "user", content: q });

    var status = document.createElement("div");
    status.className = "status";
    status.innerHTML =
      '<span class="dots"><i></i><i></i><i></i></span><span class="stxt">Searching the docs…</span>';
    var stxt = status.querySelector(".stxt");
    msgs.appendChild(status);
    msgs.scrollTop = msgs.scrollHeight;

    var answer = "";
    var body = null;
    var cutShort = false;
    // ordered trace: reasoning text interleaved with searches and page reads
    var events = [];
    var seenCalls = {};

    function setStatus(text) {
      if (status.isConnected) {
        stxt.textContent = text.length > 90 ? "…" + text.slice(-90) : text;
      }
    }

    function progress(text) {
      var last = events[events.length - 1];
      if (last && last.t === "txt") {
        last.s += text;
      } else {
        last = { t: "txt", s: text };
        events.push(last);
      }
      var lines = last.s.trim().split(/\n+/);
      var line = lines[lines.length - 1].trim();
      if (line) setStatus(line);
    }

    function onSearch(call) {
      if (!call || !call.id || seenCalls[call.id]) return;
      seenCalls[call.id] = true;
      var q, label;
      if (call.type === "store_search_call") {
        q = (call.queries || []).join(" · ");
        label = "Searching: “" + q + "”";
      } else if (call.type === "store_grep_call") {
        var p = call.pattern || "";
        if (p.indexOf("read ") === 0) {
          q = p;
          label = "Reading " + p.slice(5);
        } else {
          q = "grep " + p;
          label = "Grepping: " + p;
        }
      }
      if (!q) return;
      events.push({ t: "q", s: q });
      setStatus(label);
    }

    fetch(ENDPOINT, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages: history }),
    })
      .then(function (resp) {
        if (resp.status === 429) throw new Error("too many questions in a short time. Please wait a minute and try again.");
        if (resp.status === 403) throw new Error("this page is not allowed to use the docs chat.");
        if (!resp.ok) throw new Error("proxy returned " + resp.status);
        var reader = resp.body.getReader();
        var dec = new TextDecoder();
        var buf = "";
        function pump() {
          return reader.read().then(function (r) {
            if (r.done) return;
            buf += dec.decode(r.value, { stream: true });
            var lines = buf.split("\n");
            buf = lines.pop();
            lines.forEach(function (line) {
              if (line.indexOf("data: ") !== 0) return;
              var payload = line.slice(6);
              if (payload === "[DONE]") return;
              var chunk, delta;
              try {
                chunk = JSON.parse(payload);
              } catch (e) {
                return;
              }
              delta = (chunk.choices && chunk.choices[0] && chunk.choices[0].delta) || null;
              if (chunk.choices && chunk.choices[0] && chunk.choices[0].finish_reason === "length") cutShort = true;
              (chunk.hosted_tool_calls || []).forEach(onSearch);
              if (delta && delta.reasoning_content) progress(delta.reasoning_content);
              if (delta && delta.content) {
                if (!body) {
                  status.remove();
                  body = turn("bot", "");
                }
                answer += delta.content;
                body.innerHTML = md(answer);
                msgs.scrollTop = msgs.scrollHeight;
              }
            });
            return pump();
          });
        }
        return pump();
      })
      .then(function () {
        if (answer) {
          if (cutShort && body) {
            body.innerHTML = md(answer) + '<p class="cut">The answer hit the length limit and was cut short. Ask a follow-up for the rest.</p>';
          }
          history.push({ role: "assistant", content: answer });
          if (body && events.length) {
            var trace = document.createElement("details");
            trace.className = "trace";
            var sum = document.createElement("summary");
            var n = events.filter(function (ev) { return ev.t === "q"; }).length;
            sum.textContent = "Sources" + (n ? " · " + n + " search" + (n > 1 ? "es" : "") : "");
            trace.appendChild(sum);
            events.forEach(function (ev) {
              var row = document.createElement("div");
              if (ev.t === "q") {
                row.className = "trace-q";
                row.textContent = ev.s;
              } else {
                var text = ev.s.trim().replace(/\n{3,}/g, "\n\n");
                if (!text) return;
                row.className = "trace-body";
                row.textContent = text;
              }
              trace.appendChild(row);
            });
            body.parentNode.appendChild(trace);
          }
        } else {
          status.remove();
          turn("bot", "No answer came back. Please try again.");
        }
      })
      .catch(function (err) {
        status.remove();
        turn("bot", "**Chat is unavailable:** " + err.message);
      })
      .finally(function () {
        wrap.classList.remove("busy");
        syncInput();
        input.focus();
      });
  }

  msgs.addEventListener("click", function (e) {
    var btn = e.target.closest && e.target.closest(".copy");
    if (!btn) return;
    var pre = btn.parentNode.nextElementSibling;
    var text = pre ? pre.textContent : "";
    function done() {
      btn.classList.add("ok");
      btn.innerHTML = CHECK_ICON;
      setTimeout(function () { btn.classList.remove("ok"); btn.innerHTML = COPY_ICON; }, 1200);
    }
    if (navigator.clipboard && navigator.clipboard.writeText) navigator.clipboard.writeText(text).then(done, done);
    else done();
  });

  send.addEventListener("click", askToast);
  input.addEventListener("keydown", function (e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      askToast();
    }
  });

  function mount() {
    document.body.appendChild(host);
    watchTheme();
    var m = /[#&]ask=([^&]+)/.exec(location.hash);
    if (m) {
      setOpen(true);
      input.value = decodeURIComponent(m[1]);
      syncInput();
      setTimeout(askToast, 300);
    }
  }
  if (document.body) mount();
  else document.addEventListener("DOMContentLoaded", mount);
})();
