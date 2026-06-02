import { useState } from "react";
import { Highlight, themes } from "prism-react-renderer";

export function CodeView({ title, code }: { title: string; code: string }) {
  const [copied, setCopied] = useState(false);

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(code);
      setCopied(true);
      setTimeout(() => setCopied(false), 1200);
    } catch {
      /* clipboard unavailable (e.g. non-secure context) — ignore */
    }
  };

  return (
    <div className="codeview">
      <div className="codeview__bar">
        <span className="codeview__title">{title}</span>
        <button className="codeview__copy" onClick={copy} type="button">
          {copied ? "copied" : "copy"}
        </button>
      </div>
      <Highlight theme={themes.vsLight} code={code} language="python">
        {({ className, style, tokens, getLineProps, getTokenProps }) => (
          <pre className={`codeview__pre ${className}`} style={style}>
            {tokens.map((line, i) => {
              const lineProps = getLineProps({ line });
              return (
                <div key={i} {...lineProps}>
                  <span className="codeview__lineno">{i + 1}</span>
                  {line.map((token, key) => {
                    const tokenProps = getTokenProps({ token });
                    return <span key={key} {...tokenProps} />;
                  })}
                </div>
              );
            })}
          </pre>
        )}
      </Highlight>
    </div>
  );
}
