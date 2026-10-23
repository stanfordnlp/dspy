// "Copy page" split-button (see overrides/partials/actions.html).
//
// Copying and the Markdown link both work off a plain `index.md` sibling of
// the current page - the llmstxt plugin writes one next to every page's
// index.html at build time, so no per-page URL needs to be computed here.

(function () {
  function forEachRoot(fn) {
    document.querySelectorAll('[data-md-component="copy-page"]').forEach(fn);
  }

  function setLabel(root, text, duration) {
    var label = root.querySelector(".md-copy-page__label");
    var original = label.textContent;
    label.textContent = text;
    setTimeout(function () {
      label.textContent = original;
    }, duration);
  }

  function copyPageMarkdown(root) {
    fetch("index.md")
      .then(function (res) {
        return res.text();
      })
      .then(function (text) {
        return navigator.clipboard.writeText(text);
      })
      .then(function () {
        setLabel(root, "Copied!", 1500);
      })
      .catch(function (err) {
        console.error("Failed to copy page as Markdown:", err);
        setLabel(root, "Failed to copy", 1500);
      });
  }

  function closeMenu(root) {
    var menu = root.querySelector(".md-copy-page__menu");
    var chevron = root.querySelector(".md-copy-page__chevron");
    menu.hidden = true;
    chevron.setAttribute("aria-expanded", "false");
  }

  function openMenu(root) {
    var menu = root.querySelector(".md-copy-page__menu");
    var chevron = root.querySelector(".md-copy-page__chevron");
    menu.hidden = false;
    chevron.setAttribute("aria-expanded", "true");
  }

  forEachRoot(function (root) {
    var chevron = root.querySelector(".md-copy-page__chevron");

    root.querySelectorAll('[data-md-copy-page-action="copy"]').forEach(function (el) {
      el.addEventListener("click", function () {
        copyPageMarkdown(root);
        closeMenu(root);
      });
    });

    chevron.addEventListener("click", function () {
      var menu = root.querySelector(".md-copy-page__menu");
      if (menu.hidden) {
        openMenu(root);
      } else {
        closeMenu(root);
      }
    });
  });

  document.addEventListener("click", function (event) {
    forEachRoot(function (root) {
      if (!root.contains(event.target)) {
        closeMenu(root);
      }
    });
  });

  document.addEventListener("keydown", function (event) {
    if (event.key === "Escape") {
      forEachRoot(closeMenu);
    }
  });
})();
