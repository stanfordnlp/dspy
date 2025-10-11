document.addEventListener('DOMContentLoaded', function () {
  // Only run on the tutorials page
  const isTutorialsPage = window.location.pathname.includes('/tutorials');

  if (isTutorialsPage) {
    collapseTutorialNav();
  }

  function collapseTutorialNav() {
    // Find the navigation sidebar
    const navSidebar = document.querySelector('.md-sidebar--primary');
    if (!navSidebar) return;

    // Find the 'Tutorials' section in the navigation
    const tutorialsSection = Array.from(
      navSidebar.querySelectorAll('.md-nav__item')
    ).find((item) => {
      const linkSpan = item.querySelector('.md-nav__link .md-ellipsis');
      return linkSpan && linkSpan.textContent.trim() === 'Tutorials';
    });

    if (!tutorialsSection) return;

    // Find all nested subsections under Tutorials (level 2)
    const tutorialsNav = tutorialsSection.querySelector(
      '.md-nav[data-md-level="1"]'
    );
    if (!tutorialsNav) return;

    const subsections = tutorialsNav.querySelectorAll(
      ':scope > .md-nav__list > .md-nav__item.md-nav__item--nested'
    );

    subsections.forEach((subsection) => {
      // Find the nested navigation (level 2)
      const nestedNav = subsection.querySelector('.md-nav[data-md-level="2"]');
      if (!nestedNav) return;

      const nestedList = nestedNav.querySelector('.md-nav__list');
      if (!nestedList) return;

      const items = Array.from(
        nestedList.querySelectorAll(':scope > .md-nav__item')
      );

      // Limit to 3 items visible
      const maxVisibleItems = 3;

      if (items.length > maxVisibleItems) {
        // Hide items beyond the limit
        items.slice(maxVisibleItems).forEach((item) => {
          item.style.display = 'none';
        });

        // Get the category link to determine the 'More tutorials' URL
        const categoryLink = subsection.querySelector(
          ':scope > .md-nav__container > a.md-nav__link, :scope > a.md-nav__link'
        );
        const categoryUrl = categoryLink
          ? categoryLink.getAttribute('href')
          : '#';

        // Create and add 'More tutorials' link
        const moreTutorialsItem = document.createElement('li');
        moreTutorialsItem.className = 'md-nav__item learn-more-item';

        const moreTutorialsLink = document.createElement('a');
        moreTutorialsLink.className = 'md-nav__link';
        moreTutorialsLink.href = categoryUrl;
        moreTutorialsLink.textContent = 'More tutorials →';

        moreTutorialsItem.appendChild(moreTutorialsLink);
        nestedList.appendChild(moreTutorialsItem);
      }
    });
  }
});
