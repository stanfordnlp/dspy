import React, { useState, useEffect } from 'react';
import axios from "axios";
import styles from './display.module.css';
import { Typography, Accordion, AccordionSummary, AccordionDetails } from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';


const Display = () => {
  const [data, setData] = useState([]);
  const queryParams = new URLSearchParams(window.location.search);
  const id = queryParams.get('id');

  useEffect(() => {
    axios
    .get(`http://127.0.0.1:5000/data/${id}`)
    .then((response) => {
      setData(response.data.content);
    })
    .catch((error) => {
      console.error(error);
    });
  }, []);

  function renderItems(items) {
    return (
      <div>
        {Array.isArray(items) ?
          (items.length > 1 ?
            <div>
              {items.map((item, i) => {
                return (
                  <div key={item.id}>
                    <Typography sx={{ textAlign: 'left' }}>({i+1})</Typography>
                    {renderItems(item)}
                  </div>
                )
              })}
            </div>
            :
            renderItems(items[0])
          )
          :
          (typeof items === 'object' ? (
            <ul>
              {Object.keys(items).map((keyName, i) => (
                <Accordion
                  sx={{
                    margin: 0,
                    backgroundColor: "rgb(224, 232, 240)",
                    border: "1px solid black"
                  }}
                  key={i}
                >
                  <AccordionSummary
                    expandIcon={<ExpandMoreIcon />}
                    aria-controls="panel1a-content"
                    id="panel1a-header"
                  >
                    <Typography>{keyName}</Typography>
                  </AccordionSummary>
                  <AccordionDetails sx={{
                  }}>
                    {renderItems(items[keyName])}
                  </AccordionDetails>
                </Accordion>
              ))}
            </ul>
          )
          :
          (
            <div>
              {items}
            </div>
          )
        )
        }
      </div>
    );
  }
  
  return (
    <div className={styles.Display}>
      <ul>
        {renderItems(data)}
      </ul>
    </div>
  )
};

export default Display;
