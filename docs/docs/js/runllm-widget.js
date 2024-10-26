document.addEventListener("DOMContentLoaded", function () {
    var script = document.createElement("script");
    script.defer = true;
    script.type = "module";
    script.id = "runllm-widget-script";
    script.src =
      "https://widget.runllm.com";
    script.setAttribute("runllm-name", "DSPy");
    script.setAttribute("runllm-preset", "mkdocs");
    script.setAttribute("runllm-server-address", "https://api.runllm.com");
    script.setAttribute("runllm-assistant-id", "132");
    script.setAttribute("runllm-position", "BOTTOM_RIGHT");
    script.setAttribute("runllm-keyboard-shortcut", "Mod+j");
    script.setAttribute(
      "runllm-slack-community-url",
      ""
    );
  
    document.head.appendChild(script);
  });