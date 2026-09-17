const desktop = window.matchMedia("(min-width: 60em)");
let placeholder;

function showPlaceholder() {
    if (!desktop.matches) {
        removePlaceholder();
        return;
    }
    if (placeholder?.isConnected) return;

    const search = document.querySelector(".md-search");
    if (!search) return;

    const segment = location.pathname.split("/").filter(Boolean)[0];
    placeholder = document.createElement("div");
    placeholder.className = "md-version md-version--placeholder";
    placeholder.setAttribute("aria-hidden", "true");

    const label = document.createElement("button");
    label.type = "button";
    label.tabIndex = -1;
    label.className = "md-version__current";
    label.textContent = segment && segment !== "current" ? segment : "Current";
    placeholder.append(label);
    search.after(placeholder);
}

function removePlaceholder() {
    placeholder?.remove();
}

function placeVersionPicker() {
    const picker = document.querySelector(".md-version:not(.md-version--placeholder)");
    const search = document.querySelector(".md-search");
    const title = document.querySelector(".md-header__topic");
    if (!picker || !search || !title) {
        showPlaceholder();
        return false;
    }

    if (desktop.matches) {
        search.after(picker);
    } else {
        title.append(picker);
    }
    removePlaceholder();
    return true;
}

if (!placeVersionPicker()) {
    const observer = new MutationObserver(() => {
        if (placeVersionPicker()) observer.disconnect();
    });
    observer.observe(document.body, { childList: true, subtree: true });
}

desktop.addEventListener("change", placeVersionPicker);
