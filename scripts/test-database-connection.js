#!/usr/bin/env node

/**
 * Database Connection Test Script
 * Run this to verify your Neon database is set up correctly for Lab 2
 * 
 * Usage: node scripts/test-database-connection.js
 */

require('dotenv').config();
const { neon } = require('@neondatabase/serverless');

async function testDatabaseConnection() {
  console.log('🔍 Testing Neon Database Connection...\n');

  // Check if DATABASE_URL exists
  if (!process.env.DATABASE_URL) {
    console.error('❌ DATABASE_URL not found in .env file');
    console.log('   Add your Neon connection string to .env:');
    console.log('   DATABASE_URL="postgresql://username:password@host/database?sslmode=require"');
    process.exit(1);
  }

  try {
    const sql = neon(process.env.DATABASE_URL);

    // Test basic connection
    console.log('1. Testing basic connection...');
    const connectionTest = await sql`SELECT 'Connected!' as status, NOW() as timestamp`;
    console.log('   ✅ Database connection successful');
    console.log(`   📅 Server time: ${connectionTest[0].timestamp}\n`);

    // Check if required tables exist
    console.log('2. Checking required tables...');
    const tables = await sql`
      SELECT table_name 
      FROM information_schema.tables 
      WHERE table_schema = 'public' 
      ORDER BY table_name
    `;
    
    const tableNames = tables.map(t => t.table_name);
    const requiredTables = ['businesses', 'appointments', 'ai_metrics'];
    
    console.log(`   📋 Found tables: ${tableNames.join(', ')}`);
    
    for (const table of requiredTables) {
      if (tableNames.includes(table)) {
        console.log(`   ✅ ${table} table exists`);
      } else {
        console.log(`   ❌ ${table} table missing`);
        console.log(`      Run the SQL scripts to create missing tables`);
      }
    }

    // Test ai_metrics table structure
    if (tableNames.includes('ai_metrics')) {
      console.log('\n3. Checking ai_metrics table structure...');
      const columns = await sql`
        SELECT column_name, data_type 
        FROM information_schema.columns 
        WHERE table_name = 'ai_metrics' 
        ORDER BY ordinal_position
      `;
      
      const requiredColumns = [
        'id', 'business_id', 'session_id', 'response_time_ms', 
        'tokens_used', 'api_cost_usd', 'intent_detected'
      ];
      
      const foundColumns = columns.map(c => c.column_name);
      
      for (const col of requiredColumns) {
        if (foundColumns.includes(col)) {
          console.log(`   ✅ ${col} column exists`);
        } else {
          console.log(`   ❌ ${col} column missing`);
        }
      }
    }

    // Test inserting and reading data
    console.log('\n4. Testing data operations...');
    
    // Test businesses table
    if (tableNames.includes('businesses')) {
      try {
        const testBusiness = await sql`
          INSERT INTO businesses (name, industry, description) 
          VALUES ('Test Business', 'Technology', 'Database connection test')
          ON CONFLICT DO NOTHING
          RETURNING id, name
        `;
        
        if (testBusiness.length > 0) {
          console.log('   ✅ Can insert into businesses table');
          
          // Clean up test data
          await sql`DELETE FROM businesses WHERE name = 'Test Business'`;
          console.log('   🧹 Test data cleaned up');
        }
      } catch (error) {
        console.log('   ⚠️  Insert test failed:', error.message);
      }
    }

    console.log('\n🎉 Database setup verification complete!');
    console.log('✅ Your database is ready for Lab 2');
    
  } catch (error) {
    console.error('\n❌ Database connection failed:');
    console.error('   Error:', error.message);
    console.log('\n🔧 Troubleshooting:');
    console.log('   1. Check your DATABASE_URL format');
    console.log('   2. Ensure it includes ?sslmode=require');
    console.log('   3. Verify your Neon database is active');
    console.log('   4. Run the SQL scripts to create tables');
    process.exit(1);
  }
}

// Run the test
testDatabaseConnection().catch(console.error);