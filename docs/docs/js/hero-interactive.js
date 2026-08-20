(function () {
  "use strict";

  var state = { lm: 0, field: 0, agent: false };

  function k(t) { return '<span class="tok-k">' + t + "</span>"; }
  function s(t) { return '<span class="tok-s">' + t + "</span>"; }
  function m(t) { return '<span class="tok-m">' + t + "</span>"; }
  function tp(t) { return '<span class="tok-t">' + t + "</span>"; }
  function cm(t) { return '<span class="tok-cm">' + t + "</span>"; }
  function num(t) { return '<span class="tok-num">' + t + "</span>"; }

  var lmConfigs = [
    { line: "lm = dspy." + m("LM") + "(" + s('"openai/gpt-5.4-nano"') + ")" },
    { line: "lm = dspy." + m("LM") + "(" + s('"anthropic/claude-sonnet-4-20250514"') + ")" },
    { line: "lm = dspy." + m("LM") + "(" + s('"openrouter/deepseek/deepseek-r1"') + ")" },
    { line: "lm = dspy." + m("LM") + "(" + s('"ollama_chat/llama3.2"') + ", api_base=" + s('"http://localhost:11434"') + ")" },
  ];

  var fieldConfigs = [
    {
      name: "location",
      code: "    location: " + tp("str") + " = dspy." + m("OutputField") + "()",
      output: "  location=" + s('"Building 4, Room 201"'),
      isInput: false,
    },
    {
      name: "rsvp_required",
      code: "    rsvp_required: " + tp("bool") + " = dspy." + m("OutputField") + "()",
      output: "  rsvp_required=" + num("True"),
      isInput: false,
    },
    {
      name: "attendees",
      code: "    attendees: " + tp("list") + "[" + tp("str") + "] = dspy." + m("OutputField") + "()",
      output: "  attendees=[" + s('"Alice"') + ", " + s('"Bob"') + ", " + s('"Carol"') + "]",
      isInput: false,
    },
    {
      name: "urgency",
      code: "    urgency: " + tp("Literal") + "[" + s('"low"') + ", " + s('"medium"') + ", " + s('"high"') + "] = dspy." + m("OutputField") + "()",
      output: "  urgency=" + s('"medium"'),
      isInput: false,
    },
    {
      name: "duration_minutes",
      code: "    duration_minutes: " + tp("int") + " = dspy." + m("OutputField") + "()",
      output: "  duration_minutes=" + num("60"),
      isInput: false,
    },
    {
      name: "is_virtual",
      code: "    is_virtual: " + tp("bool") + " = dspy." + m("OutputField") + "()",
      output: "  is_virtual=" + num("False"),
      isInput: false,
    },
    {
      name: "calendar",
      code: "    calendar: " + tp("Literal") + "[" + s('"work"') + ", " + s('"personal"') + "] = dspy." + m("OutputField") + "()",
      output: "  calendar=" + s('"work"'),
      isInput: false,
    },
    {
      name: "subject",
      code: "    subject: " + tp("str") + " = dspy." + m("InputField") + "()",
      output: null,
      isInput: true,
    },
  ];

  var toolLines = [
    k("def") + " " + m("check_calendar") + "(date: " + tp("str") + ") -> " + tp("bool") + ":",
    "    " + s('"""Check for scheduling conflicts."""'),
    "    " + k("return") + " calendar.has_conflict(date)",
  ];

  var classLines = [
    k("class") + " " + tp("ExtractEvent") + "(dspy.Signature):",
    "    " + s('"""Extract event details from an email."""'),
    "    email: " + tp("str") + " = dspy." + m("InputField") + "()",
    "    event_name: " + tp("str") + " = dspy." + m("OutputField") + "()",
    "    date: " + tp("str") + " = dspy." + m("OutputField") + "()",
  ];

  function buildCodeLines() {
    var lines = [];

    lines.push(lmConfigs[state.lm].line);
    lines.push("&nbsp;");

    for (var i = 0; i < classLines.length; i++) {
      lines.push(classLines[i]);
    }

    if (state.field > 0) {
      lines.push(fieldConfigs[state.field - 1].code);
    }

    lines.push("&nbsp;");

    if (state.agent) {
      for (var j = 0; j < toolLines.length; j++) {
        lines.push(toolLines[j]);
      }
      lines.push("&nbsp;");
      lines.push("extract = dspy." + m("ReAct") + "(ExtractEvent, tools=[check_calendar])");
    } else {
      lines.push("extract = dspy." + m("Predict") + "(ExtractEvent)");
    }

    if (state.field === 8) {
      lines.push("extract(email=inbox_message, subject=" + s('"Team Offsite"') + ")");
    } else {
      lines.push("extract(email=inbox_message)");
    }

    return lines;
  }

  function buildOutputLines() {
    var lines = [];
    var fc = state.field > 0 ? fieldConfigs[state.field - 1] : null;
    var hasExtraOutput = fc && !fc.isInput;

    lines.push("Prediction(");
    lines.push("  event_name=" + s('"Team Offsite"') + ",");

    if (hasExtraOutput || state.agent) {
      lines.push("  date=" + s('"Thursday, June 5"') + ",");
    } else {
      lines.push("  date=" + s('"Thursday, June 5"'));
    }

    if (hasExtraOutput && state.agent) {
      lines.push(fc.output + ",");
      lines.push("  has_conflict=" + num("False"));
    } else if (hasExtraOutput) {
      lines.push(fc.output);
    } else if (state.agent) {
      lines.push("  has_conflict=" + num("False"));
    }

    lines.push(")");
    return lines;
  }

  function linesToHTML(lines) {
    var html = "";
    for (var i = 0; i < lines.length; i++) {
      html += '<div class="line">' + lines[i] + "</div>";
    }
    return html;
  }

  function updateLineNumbers(count) {
    var el = document.getElementById("hp-hero-linenums");
    if (!el) return;
    var html = "";
    for (var i = 1; i <= count; i++) {
      html += "<span>" + i + "</span>";
    }
    el.innerHTML = html;
  }

  function diffAndHighlight(container, newHTML, skipHeader) {
    var oldLines = container.querySelectorAll(".line");
    var oldContents = [];
    for (var i = 0; i < oldLines.length; i++) {
      oldContents.push(oldLines[i].innerHTML);
    }

    if (skipHeader) {
      var header = container.querySelector(".hp-code-output-header");
      container.innerHTML = (header ? header.outerHTML : "") + newHTML;
    } else {
      container.innerHTML = newHTML;
    }

    var newLines = container.querySelectorAll(".line");
    for (var j = 0; j < newLines.length; j++) {
      var txt = newLines[j].textContent.trim();
      if (txt && (j >= oldContents.length || newLines[j].innerHTML !== oldContents[j])) {
        newLines[j].classList.add("hp-line-flash");
      }
    }
  }

  function silentRender() {
    var codeEl = document.getElementById("hp-hero-code");
    var outputEl = document.getElementById("hp-hero-output");
    if (!codeEl || !outputEl) return;

    var codeLines = buildCodeLines();
    var outputLines = buildOutputLines();

    codeEl.innerHTML = linesToHTML(codeLines);
    var header = outputEl.querySelector(".hp-code-output-header");
    outputEl.innerHTML = (header ? header.outerHTML : "") + linesToHTML(outputLines);
    updateLineNumbers(codeLines.length);
    updateButtons();
  }

  function render() {
    var codeEl = document.getElementById("hp-hero-code");
    var outputEl = document.getElementById("hp-hero-output");
    if (!codeEl || !outputEl) return;

    var codeLines = buildCodeLines();
    var outputLines = buildOutputLines();

    diffAndHighlight(codeEl, linesToHTML(codeLines), false);
    diffAndHighlight(outputEl, linesToHTML(outputLines), true);
    updateLineNumbers(codeLines.length);
    updateButtons();
  }

  function updateButtons() {
    var lmBtn = document.getElementById("hp-btn-lm");
    var fieldBtn = document.getElementById("hp-btn-field");
    var agentBtn = document.getElementById("hp-btn-agent");
    if (!lmBtn || !fieldBtn || !agentBtn) return;

    lmBtn.classList.remove("active");

    if (state.field === 0) {
      fieldBtn.textContent = "Add a Field";
      fieldBtn.classList.remove("active");
    } else {
      fieldBtn.textContent = "Swap a Field";
      fieldBtn.classList.add("active");
    }

    if (state.agent) {
      agentBtn.classList.add("active");
    } else {
      agentBtn.classList.remove("active");
    }
  }

  function track(eventName, params) {
    if (typeof window.gtag === "function") {
      window.gtag("event", eventName, params || {});
    }
  }

  function init() {
    var lmBtn = document.getElementById("hp-btn-lm");
    var fieldBtn = document.getElementById("hp-btn-field");
    var agentBtn = document.getElementById("hp-btn-agent");
    if (lmBtn && fieldBtn && agentBtn) {
      lmBtn.addEventListener("click", function () {
        state.lm = (state.lm + 1) % lmConfigs.length;
        render();
        track("hero_code_button_click", {
          button: "change_llm",
          lm_index: state.lm,
          lm_provider: ["openai", "anthropic", "openrouter", "ollama"][state.lm],
        });
      });

      fieldBtn.addEventListener("click", function () {
        state.field = (state.field + 1) % (fieldConfigs.length + 1);
        render();
        var fieldName = state.field === 0 ? "removed" : fieldConfigs[state.field - 1].name;
        track("hero_code_button_click", {
          button: "add_field",
          field_index: state.field,
          field_name: fieldName,
        });
      });

      agentBtn.addEventListener("click", function () {
        state.agent = !state.agent;
        render();
        track("hero_code_button_click", {
          button: "make_agent",
          agent_enabled: state.agent,
        });
      });

      silentRender();
    }

    var tabIds = ["extract", "agent", "pipeline", "multimodal", "optimize"];
    tabIds.forEach(function (id) {
      var label = document.querySelector('label[for="hp-tab-' + id + '"]');
      if (label) {
        label.addEventListener("click", function () {
          track("example_tab_click", { tab: id });
        });
      }
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
