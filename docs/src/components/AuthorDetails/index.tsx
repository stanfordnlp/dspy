import clsx from 'clsx';
import Heading from '@theme/Heading';
import Link from '@docusaurus/Link';
import styles from './styles.module.css';

type FeatureItem = {
  name: string;
  twitterLink?: string;
  linkedinLink?: string;
  portfolioLink?: string;
};

const FeatureList: FeatureItem[] = [
  {
    name: 'Herumb Shandilya',
    twitterLink: 'https://twitter.com/krypticmouse',
    linkedinLink: 'https://www.linkedin.com/in/herumb-shandilya/',
    portfolioLink: 'https://herumbshandilya.com/',
  },
  {
    name: 'Arnav Singhvi',
    twitterLink: 'https://twitter.com/arnav_thebigman',
    linkedinLink: 'https://www.linkedin.com/in/arnav-singhvi-1323a7189/',
    portfolioLink: 'https://arnavsinghvi11.github.io/',
  },
  {
    name: 'Prajapati Harishkumar Kishorkumar'
  }
];

export default function AuthorDetails(props: FeatureItem): JSX.Element {
  if (props.name && !props.twitterLink && !props.bio && !props.img) {
    const author = FeatureList.find((i) => i.name === props.name);
    if (!author) {
      throw new Error(`Author not found for name: ${props.name}.`);
    }
    props = author;
  }

  return (
    <div>
      <div>
        <p style={{ fontSize: "1.2rem", margin: 1, paddingBottom:"0.4rem" }}><span style={{ fontWeight: "600" }}>Written By:</span> {props.name}</p>
        {props.twitterLink &&
          <Link href={props.twitterLink}>
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512" width="25px" height="25px" className={clsx(styles.svg)}>
              <path d="M389.2 48h70.6L305.6 224.2 487 464H345L233.7 318.6 106.5 464H35.8L200.7 275.5 26.8 48H172.4L272.9 180.9 389.2 48zM364.4 421.8h39.1L151.1 88h-42L364.4 421.8z" />
            </svg>
          </Link>
        }
        {props.linkedinLink &&
          <Link href={props.linkedinLink}>
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 448 512" width="25px" height="25px" className={clsx(styles.svg)} style={{ marginLeft:"0.8rem" }}>
              <path d="M100.3 448H7.4V148.9h92.9zM53.8 108.1C24.1 108.1 0 83.5 0 53.8a53.8 53.8 0 0 1 107.6 0c0 29.7-24.1 54.3-53.8 54.3zM447.9 448h-92.7V302.4c0-34.7-.7-79.2-48.3-79.2-48.3 0-55.7 37.7-55.7 76.7V448h-92.8V148.9h89.1v40.8h1.3c12.4-23.5 42.7-48.3 87.9-48.3 94 0 111.3 61.9 111.3 142.3V448z"/>
            </svg>
          </Link>
        }
        {props.portfolioLink &&
          <Link href={props.portfolioLink}>
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512" width="25px" height="25px" className={clsx(styles.svg)} style={{ marginLeft:"0.8rem" }}>
              <path d="M352 256c0 22.2-1.2 43.6-3.3 64H163.3c-2.2-20.4-3.3-41.8-3.3-64s1.2-43.6 3.3-64H348.7c2.2 20.4 3.3 41.8 3.3 64zm28.8-64H503.9c5.3 20.5 8.1 41.9 8.1 64s-2.8 43.5-8.1 64H380.8c2.1-20.6 3.2-42 3.2-64s-1.1-43.4-3.2-64zm112.6-32H376.7c-10-63.9-29.8-117.4-55.3-151.6c78.3 20.7 142 77.5 171.9 151.6zm-149.1 0H167.7c6.1-36.4 15.5-68.6 27-94.7c10.5-23.6 22.2-40.7 33.5-51.5C239.4 3.2 248.7 0 256 0s16.6 3.2 27.8 13.8c11.3 10.8 23 27.9 33.5 51.5c11.6 26 20.9 58.2 27 94.7zm-209 0H18.6C48.6 85.9 112.2 29.1 190.6 8.4C165.1 42.6 145.3 96.1 135.3 160zM8.1 192H131.2c-2.1 20.6-3.2 42-3.2 64s1.1 43.4 3.2 64H8.1C2.8 299.5 0 278.1 0 256s2.8-43.5 8.1-64zM194.7 446.6c-11.6-26-20.9-58.2-27-94.6H344.3c-6.1 36.4-15.5 68.6-27 94.6c-10.5 23.6-22.2 40.7-33.5 51.5C272.6 508.8 263.3 512 256 512s-16.6-3.2-27.8-13.8c-11.3-10.8-23-27.9-33.5-51.5zM135.3 352c10 63.9 29.8 117.4 55.3 151.6C112.2 482.9 48.6 426.1 18.6 352H135.3zm358.1 0c-30 74.1-93.6 130.9-171.9 151.6c25.5-34.2 45.2-87.7 55.3-151.6H493.4z"/>
            </svg>
          </Link>
        }
      </div>
    </div>
  );
}
