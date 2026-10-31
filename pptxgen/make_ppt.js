const PptxGenJS = require('pptxgenjs');
const fs = require('fs');
const path = require('path');

const pres = new PptxGenJS();
pres.layout = 'LAYOUT_WIDE';
pres.author = 'Majority-Rule ABC Study';
pres.title = 'Influence Maximisation under the Majority-Rule Diffusion Model';
pres.subject = 'Updated deck: monotone, not submodular; ABC optimiser; equal-budget evaluation';

const PW = 13.333;
const PH = 7.5;

const screenDir = path.join(__dirname, '..', 'deck', 'screens');
const files = fs.readdirSync(screenDir)
  .filter((f) => /^slide-\d+\.png$/i.test(f))
  .sort((a, b) => {
    const an = parseInt(a.match(/\d+/)[0], 10);
    const bn = parseInt(b.match(/\d+/)[0], 10);
    return an - bn;
  });

if (files.length === 0) {
  console.error('No slide-NN.png files found in deck/screens/');
  process.exit(1);
}

files.forEach((file, i) => {
  const slide = pres.addSlide();
  slide.background = { color: 'FFFFFF' };
  slide.addImage({
    path: path.join(screenDir, file),
    x: 0,
    y: 0,
    w: PW,
    h: PH,
    sizing: { type: 'contain', w: PW, h: PH },
  });
});

const outPath = path.join(__dirname, '..', 'Majority_Rule_ABC_Updated_Deck.pptx');
pres.writeFile({ fileName: outPath })
  .then(() => {
    const stats = fs.statSync(outPath);
    console.log(`Wrote ${outPath}`);
    console.log(`Slides: ${files.length} | Size: ${(stats.size / 1024 / 1024).toFixed(2)} MB`);
  })
  .catch((err) => {
    console.error(err);
    process.exit(1);
  });
