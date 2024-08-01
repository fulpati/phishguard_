const express = require('express');
const Email = require('../models/Email');
const { checkPhishing } = require('../utils/phishingDetection');

const router = express.Router();

router.post('/send', async (req, res) => {
  const { sender, receiver, subject, body } = req.body;
  const isPhishing = await checkPhishing(body);

  const email = new Email({
    sender,
    receiver,
    subject,
    body,
    isPhishing,
    folder: isPhishing ? 'spam' : 'inbox',
  });
  
  await email.save();
  res.status(201).send('Email sent');
});

router.get('/inbox', async (req, res) => {
  const { userId } = req.query;
  const emails = await Email.find({ receiver: userId, folder: 'inbox' });
  res.status(200).json(emails);
});

router.get('/spam', async (req, res) => {
  const { userId } = req.query;
  const emails = await Email.find({ receiver: userId, folder: 'spam' });
  res.status(200).json(emails);
});

module.exports = router;

