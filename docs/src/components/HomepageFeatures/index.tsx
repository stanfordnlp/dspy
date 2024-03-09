import clsx from 'clsx';
import Heading from '@theme/Heading';
import styles from './styles.module.css';

type FeatureItem = {
  title: string;
  img: string;
  description: JSX.Element;
};

const FeatureList: FeatureItem[] = [
  {
    title: 'Systematic Optimization',
    img: '/img/optimize.png',
    description: (
      <>
        Choose from a range of optimizers to enhance your program. Whether it's generating refined instructions, or fine-tuning weights, DSPy's optimizers are engineered to maximize efficiency and effectiveness.
      </>
    ),
  },
  {
    title: 'Modular Approach',
    img: '/img/modular.png',
    description: (
      <>
        With DSPy, you can build your system using predefined modules, replacing intricate prompting techniques with straightforward, effective solutions.
      </>
    ),
  },
  {
    title: 'Cross-LM Compatibility',
    img: '/img/universal_compatibility.png',
    description: (
      <>
        Whether you're working with powerhouse models like GPT-3.5 or GPT-4, or local models such as T5-base or Llama2-13b, DSPy seamlessly integrates and enhances their performance in your system.
      </>
    ),
  },
];

function Feature({title, img, description}: FeatureItem) {
  return (
    <div className={clsx('col col--4')}>
      <div className="text--center">
        <img className={styles.featureSvg} src={img} alt={title}  style={{objectFit: 'cover'}} />
      </div>
      <div className="text--center padding-horiz--md">
        <Heading as="h3">{title}</Heading>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures(): JSX.Element {
  return (
    <section className={styles.features} style={{marginBottom: "2rem"}}>
      <div className="container">
        <p style={{color:`var(--hero-text-color)`, fontWeight:"700", fontSize: "2rem", textAlign: "center"}}>The Way of DSPy</p>
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}
