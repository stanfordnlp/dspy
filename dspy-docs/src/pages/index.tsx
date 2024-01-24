import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import HomepageFeatures from '@site/src/components/HomepageFeatures';
import Heading from '@theme/Heading';

import styles from './index.module.css';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <>
      <header className={clsx('hero', styles.heroBanner)} style={{backgroundColor:`var(--bg-color)`}}>
        <div className="container">
          <div style={{display: 'flex', alignItems: 'center', justifyContent: 'center'}}>
            <img src="/img/logo.png" alt="DSPy Logo" width="150rem"/>
            <div style={{color:`var(--hero-text-color)`,fontSize:"10rem",fontWeight:"600", marginLeft: '1rem'}}>
              {siteConfig.title}
            </div>
          </div>
          <p style={{color:`var(--hero-text-color)`, fontWeight:"500", fontSize: "2rem"}}>{siteConfig.tagline}</p>
          <div className={styles.buttons}>
            <Link
              className={clsx('button button--secondary button--lg', styles.buttonHoverEffect)}
              to="/docs/intro"
            >
              Get Started with DSPy
            </Link>
          </div>
        </div>
      </header>
      <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1440 130">
        <path fill={`var(--bg-color)`} fill-opacity="1" d="M0,128L120,112C240,96,480,64,720,58.7C960,53,1200,75,1320,85.3L1440,96L1440,0L1320,0C1200,0,960,0,720,0C480,0,240,0,120,0L0,0Z"></path>
      </svg>
    </>
  );
}

export default function Home(): JSX.Element {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={`${siteConfig.title} Documentation`}
      description="Programming—not prompting—Foundation Models">
      <HomepageHeader />
      <main>
        <HomepageFeatures />
      </main>
    </Layout>
  );
}
