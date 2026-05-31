const celery = require('celery-node');

console.log("BROKER:", process.env.CELERY_BROKER_URL);
console.log("RESULT:", process.env.CELERY_RESULT_BACKEND);

const client = celery.createClient(
  process.env.CELERY_BROKER_URL,
  process.env.CELERY_RESULT_BACKEND
);

module.exports = client;
