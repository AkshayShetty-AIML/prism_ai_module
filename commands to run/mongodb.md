# MongoDB — Start & Inspect Commands

## Start MongoDB

### Homebrew (install once, then start)
```bash
brew tap mongodb/brew
brew install mongodb-community
brew services start mongodb-community
```

### Stop / Restart
```bash
brew services stop mongodb-community
brew services restart mongodb-community
```

### Docker (alternative — no install needed)
```bash
docker run -d --name mongodb -p 27017:27017 mongo:7
docker stop mongodb
docker start mongodb
```

---

## Connect with mongosh

```bash
mongosh
```

### Switch to the PRISM database
```js
use prism_db2
```

---

## Inspect Records

### Count all analyzed records
```js
db.analyzed_records.countDocuments()
```

### View a single record (full)
```js
db.analyzed_records.findOne()
```

### View all records (key fields only)
```js
db.analyzed_records.find(
  {},
  { item_id: 1, sentiment: 1, bot_flag: 1, credibility_tier: 1, pipeline_stage_stopped: 1, _id: 0 }
).pretty()
```

### Filter by sentiment
```js
db.analyzed_records.find({ sentiment: "positive" }).pretty()
db.analyzed_records.find({ sentiment: "negative" }).pretty()
```

### Filter by bot flag
```js
db.analyzed_records.find({ bot_flag: "bot" }).pretty()
db.analyzed_records.find({ bot_flag: "human" }).pretty()
```

### Filter by pipeline stage (e.g. filtered out by noise filter)
```js
db.analyzed_records.find({ pipeline_stage_stopped: "noise_filter" }).pretty()
db.analyzed_records.find({ pipeline_stage_stopped: "complete" }).pretty()
```

### Filter by batch
```js
db.analyzed_records.find({ batch_id: "test-batch-001" }).pretty()
```

---

## Inspect Batch Jobs

```js
db.batch_jobs.countDocuments()
db.batch_jobs.find().pretty()
db.batch_jobs.find({ status: "failed" }).pretty()
```

---

## Wipe Collections (for re-testing)

```js
db.analyzed_records.deleteMany({})
db.batch_jobs.deleteMany({})
```

---

## Python Quick Check (no mongosh needed)

```bash
python3 -c "
import asyncio
from motor.motor_asyncio import AsyncIOMotorClient

async def check():
    client = AsyncIOMotorClient('mongodb://localhost:27017')
    db = client['prism_db2']
    count = await db.analyzed_records.count_documents({})
    print(f'{count} records in analyzed_records')
    doc = await db.analyzed_records.find_one()
    if doc:
        fields = ['item_id', 'sentiment', 'bot_flag', 'credibility_tier', 'pipeline_stage_stopped']
        print({k: doc[k] for k in fields if k in doc})

asyncio.run(check())
"
```
