// Importing necessary modules from Docusaurus and Prism for syntax highlighting
import { themes as prismThemes } from 'prism-react-renderer';
import type { Config } from '@docusaurus/types';

// Config object type declaration for TypeScript
const config: Config = {
  title: 'DSPy',
  tagline: 'Programming—not prompting—Language Models',
  favicon: 'img/logo.png',
  
  // The URL and base URL for your project
  url: 'https://dspy.ai',
  baseUrl: '/',
  
  // GitHub configuration for deployment and organization information
  organizationName: 'stanfordnlp',
  projectName: 'dspy',

  // Handling of broken links and markdown links
  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',

  // Internationalization configuration, with 'en' as the default language
  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  // Preset configuration using the 'classic' preset
  presets: [
    [
      'classic',
      {
        // Configuration for the docs portion of the site
        docs: {
          path: './docs', // Path to the documentation markdown files
          routeBasePath: '/docs/', // URL route for the documentation section
          sidebarPath: require.resolve('./sidebars.ts'), // Path to the sidebar configuration
          // URL for the "edit this page" feature
          // editUrl: 'https://github.com/facebook/docusaurus/tree/main/packages/create-docusaurus/templates/shared/',
        },
        // Configuration for the blog portion of the site
        blog: {
          showReadingTime: true, // Shows estimated reading time for blog posts
          // URL for the "edit this page" feature for blog posts
          // editUrl: 'https://github.com/facebook/docusaurus/tree/main/packages/create-docusaurus/templates/shared/',
        },
        // Configuration for the site theme
        theme: {
          customCss: require.resolve('./src/css/custom.css'), // Path to custom CSS
        },
      },
    ],
  ],
  // Plugins configuration
  plugins: [
    // Additional plugin for API documentation
    [
      '@docusaurus/plugin-content-docs',
      {
        id: 'api', // Unique identifier for the API docs
        path: './api', // Path to the API documentation markdown files
        routeBasePath: '/api/', // URL route for the API documentation section
        sidebarPath: require.resolve('./sidebars.ts'), // Path to the API sidebar configuration
        // URL for the "edit this page" feature for the API docs
        // editUrl: 'https://github.com/stanfordnlp/dspy/tree/main/api',
      },
    ],
  ],
  // Theme configuration
  themeConfig: {
    image: 'img/logo.png', // Image for social sharing cards
    navbar: {
      title: 'DSPy',
      logo: {
        alt: 'DSPy Logo', // Alternate text for the logo
        src: 'img/logo.png', // Path to the logo image
      },
      // Navbar items configuration
      items: [
        {
          type: 'docSidebar', // Type of item is a doc sidebar
          sidebarId: 'tutorialSidebar', // ID of the sidebar to use
          position: 'left', // Position in the navbar
          label: 'Documentation', // Label for the navbar item
        },
        // Navbar item for the API reference, linking to the intro document
        { to: '/docs/category/tutorials', label: 'Tutorials', position: 'left' },
        // Navbar item for the API reference, linking to the intro document
        { to: '/api/intro', label: 'API References', position: 'left' },
        // Navbar item for the API reference, linking to the intro document
        { to: '/docs/cheatsheet', label: 'DSPy Cheatsheet', position: 'right' },
        // Navbar item for the GitHub repository
        {
          href: 'https://github.com/stanfordnlp/dspy',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'light', // Style of the footer
      links: [
        // Links configuration for the footer
        {
          title: 'Docs', // Title for the docs link section
          items: [
            // Individual link items for the docs
            {
              label: 'Documentation',
              to: '/docs/intro',
            },
            {
              label: 'API Reference',
              to: '/api/intro',
            },
          ],
        },
        // Links for the community section of the footer
        {
          title: 'Community',
          items: [
            {
              label: 'Omar Khattab',
              href: 'https://twitter.com/lateinteraction',
            },
            {
              label: 'Herumb Shandilya',
              href: 'https://twitter.com/krypticmouse',
            },
            {
              label: 'Arnav Singhvi',
              href: 'https://twitter.com/arnav_thebigman',
            },
          ],
        },
        // Additional links under the 'More' section
        {
          title: 'More',
          items: [
            {
              label: 'GitHub',
              href: 'https://github.com/stanfordnlp/dspy',
            },
          ],
        },
      ],
      // Copyright statement for the footer
      copyright: `Built with ⌨️`,
    },
    // Prism theme configuration for code syntax highlighting
    prism: {
      theme: prismThemes.github, // Light theme for code blocks
      darkTheme: prismThemes.dracula, // Dark theme for code blocks
    },
  },
};

// Exporting the configuration object
export default config;
